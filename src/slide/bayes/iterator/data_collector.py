import json
import logging
import os
import random
import threading
import time
from collections import defaultdict

from src.slide.bayes.LitmusPipeline import LitmusPipeline, SharedResourceManager
from src.slide.bayes.framework import LitmusRunner
from src.slide.bayes.litmus_param_space import LitmusParamSpace
from src.slide.bayes.logger_util import setup_logger, get_logger
from src.slide.bayes.util import get_files, parse_log_by_mode_perple, parse_log_by_mode
from src.slide.utils.cmd_util import run_cmd

SEED = 2025
LOG_NAME = "collector_run"


# ================= 结果缓存类 =================
class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        # 为了避免多线程写入冲突，这里只做内存记录和追加写入，不做复杂逻辑
        # 实际的合并去重交给主控脚本的 aggregator 处理
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        key = self._make_key(obj["litmus"], obj["param"])
                        self.data[key] = obj["score"]
                    except:
                        pass
        self.f = open(path, "a")
        self.lock = threading.Lock()

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        key = self._make_key(litmus, param_vec)
        with self.lock:
            if key in self.data: return
            self.data[key] = score
            self.f.write(json.dumps({
                "litmus": litmus, "param": param_vec, "score": score
            }) + "\n")
            self.f.flush()


# ================= 任务执行类 =================
class TaskBasedRunner(LitmusRunner):
    def __init__(
            self,
            shared_todo_queue,  # 共享的任务队列 [(litmus, vec), ...]
            param_space,
            stat_log,
            resource_manager,
            mode="time",
            pipeline_host="127.0.0.1",
            pipeline_user="root",
            pipeline_pass="root",
            pipeline_port=22,
            litmus_path_base="",
            litmus_dir_path="",
            remote_work_dir=""
    ):
        super().__init__([], [], stat_log, mode)  # litmus_list 传空，因为我们用队列
        self.shared_queue = shared_todo_queue
        self.ps = param_space
        self.litmus_path_base = litmus_path_base
        self.litmus_dir_path = litmus_dir_path

        self.log_dir = f"{os.path.dirname(stat_log)}/bayes_log_{pipeline_host}"
        run_cmd(f"mkdir -p {self.log_dir}")

        self.logger = setup_logger(
            log_file=stat_log,
            level=logging.INFO,
            name=f"runner_{pipeline_host}",
            stdout=True
        )
        self.resource_manager = resource_manager

        # 本地 Cache，避免重复提交
        self.cache = ResultCache(stat_log + ".cache.jsonl")

        # 初始化 Pipeline
        self.pipeline = LitmusPipeline(
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            logger=self.logger,
            resource_manager=self.resource_manager,
            remote_work_dir=remote_work_dir
        )
        self.pipeline.start(compiler_thread_count=2, downloader_thread_count=2)

    def _submit_one(self, litmus, vec):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        params._temp_vec = vec
        litmus_file = f"{self.litmus_path_base}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=self.litmus_dir_path,
            log_dir_path=self.log_dir,
            run_time=100000
        )

    def _parse_log_to_score(self, log_path, litmus_name, has_perple, mode):
        try:
            if has_perple:
                return parse_log_by_mode_perple(log_path, mode)
            else:
                return parse_log_by_mode(log_path, mode)
        except Exception as e:
            self.logger.error(f"Parse error {log_path}: {e}")
            return -1

    def run(self):
        MAX_IN_FLIGHT = 10
        active_count = 0

        def try_fill_pipeline():
            nonlocal active_count
            # 从共享队列获取任务 (线程安全)
            while active_count < MAX_IN_FLIGHT:
                try:
                    # 非阻塞获取，或者带超时
                    task_item = self.shared_queue.pop(0)  # 这是一个简单的 list pop，多线程不安全，需加锁
                except IndexError:
                    break  # 队列空了

                litmus, vec = task_item

                # Check cache
                if self.cache.get(litmus, vec) is not None:
                    continue

                self._submit_one(litmus, vec)
                active_count += 1

        # 加锁处理 list pop 的简易封装
        def safe_pop():
            # 注意：在多线程环境中直接 pop(0) 可能有竞争，Python 的 list 操作通常是原子的，但为了稳妥建议用 Queue
            # 这里为保持代码结构简单，假设外部传入的是线程安全的 Queue 或者我们在 pop 时加锁
            try:
                return self.shared_queue.pop(0)
            except IndexError:
                return None

        # 重新定义 try_fill (Thread Safe Version)
        def try_fill_pipeline_safe():
            nonlocal active_count
            while active_count < MAX_IN_FLIGHT:
                # 简单粗暴的锁 (虽然 Python GIL 会帮忙，但 list pop(0) 性能较差，这里假设 list 不大)
                # 更好的方式是使用 queue.Queue
                item = None
                try:
                    item = self.shared_queue.get_nowait()
                except:
                    break

                if item:
                    litmus, vec = item
                    if self.cache.get(litmus, vec) is not None:
                        continue
                    self._submit_one(litmus, vec)
                    active_count += 1

        try_fill_pipeline_safe()

        if active_count == 0 and self.shared_queue.empty():
            self.logger.info("Nothing to do.")
            self.pipeline.wait_completion()
            return

        for result in self.pipeline.stream_results():
            active_count -= 1
            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            if hasattr(task.params, '_temp_vec'):
                score = self._parse_log_to_score(log_path, litmus_name, task.params.is_perple(), self.mode)
                self.logger.info(f"[DONE] {litmus_name} | Score: {score:.2f}")
                self.cache.add(litmus_name, task.params._temp_vec, score)

            try_fill_pipeline_safe()

            if self.shared_queue.empty() and active_count == 0:
                break

        self.pipeline.wait_completion()


# ================= 入口函数 =================

from queue import Queue


def main_collector(task_list, board_configs, base_config, iter):
    """
    Args:
        task_list: list of dict {'litmus': name, 'param': vec}
        board_configs: list of dict (host, user, pass...)
        base_config: dict (litmus_path_base, stat_log_dir, ...)
    """
    random.seed(SEED)
    ts = time.strftime("%Y%m%d-%H%M%S")
    logger = setup_logger(
        log_file=f"{base_config['stat_log_dir']}/collector_master.{ts}.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )

    logger.info(f"=== [Collector] Start | Tasks: {len(task_list)} | Boards: {len(board_configs)} ===")

    # 1. 构建共享任务队列
    # 使用 Queue 保证多线程安全
    task_queue = Queue()
    for t in task_list:
        task_queue.put((t['litmus'], t['param']))

    logger.info("Task queue populated.")

    param_space = LitmusParamSpace()
    global_manager = SharedResourceManager(total_consumers=len(board_configs))

    threads = []

    for board in board_configs:
        # 每个板子单独的 stat log
        board_stat_log = f"{base_config['stat_log_dir']}/log_record_{board['host']}-{iter}.log"

        runner = TaskBasedRunner(
            shared_todo_queue=task_queue,
            param_space=param_space,
            stat_log=board_stat_log,
            resource_manager=global_manager,
            pipeline_host=board['host'],
            pipeline_user=board['user'],
            pipeline_pass=board['pass'],
            pipeline_port=board['port'],
            litmus_path_base=base_config['litmus_path_base'],
            litmus_dir_path=base_config['litmus_dir_path'],
            remote_work_dir=base_config['remote_work_dir']
        )

        t = threading.Thread(target=runner.run)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logger.info("All boards finished.")