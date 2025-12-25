import json
import logging
import os
import random
import time
from collections import defaultdict

from scipy.stats import norm

from src.slide.bayes.LitmusPipeline import LitmusPipeline
from src.slide.bayes.framework import LitmusRunner, get_score
from src.slide.bayes.litmus_param_space import LitmusParamSpace
from src.slide.bayes.litmus_params import LitmusParams
from src.slide.bayes.logger_util import setup_logger, get_logger
from src.slide.bayes.util import get_files, read_log_to_summary, parse_log_by_mode_perple, parse_log_by_mode
import torch
from sklearn.ensemble import RandomForestRegressor
import numpy as np

SEED = 2025
LOG_NAME = "bayes"
class ResultCache:
    def __init__(self, path):
        self.path = path
        self.data = {}

        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj["score"]

        # 追加写
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def get(self, litmus, param_vec):
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, score):
        key = self._make_key(litmus, param_vec)
        if key in self.data:
            return

        self.data[key] = score
        self.f.write(json.dumps({
            "litmus": litmus,
            "param": param_vec,
            "score": score
        }) + "\n")
        self.f.flush()

class ErrorCache:
    """
    Cache for failed / invalid litmus runs.
    Once a (litmus, param) is recorded here, it should never be retried.
    """

    def __init__(self, path):
        self.path = path
        self.data = {}

        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    key = self._make_key(obj["litmus"], obj["param"])
                    self.data[key] = obj

        # append-only
        self.f = open(path, "a")

    def _make_key(self, litmus, param_vec):
        return f"{litmus}|" + ",".join(map(str, param_vec))

    def has(self, litmus, param_vec):
        """
        Check whether this (litmus, param) has failed before.
        """
        return self._make_key(litmus, param_vec) in self.data

    def get(self, litmus, param_vec):
        """
        Get error record (dict) or None.
        """
        return self.data.get(self._make_key(litmus, param_vec))

    def add(self, litmus, param_vec, error_type, error_msg=None, extra=None):
        """
        Record a failed run.

        Args:
            error_type: short string, e.g. "timeout", "compile_error", "runtime_error"
            error_msg: optional detailed message
            extra: optional dict for additional info
        """
        key = self._make_key(litmus, param_vec)
        if key in self.data:
            return

        record = {
            "litmus": litmus,
            "param": param_vec,
            "error_type": error_type,
            "error_msg": error_msg,
            "extra": extra,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.data[key] = record
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

class RandomForestBO:

    def __init__(self, param_space: LitmusParamSpace, litmus_list, litmus_vec_path, n_estimators=200,
                 ):
        self.ps = param_space
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=1,
            max_features="sqrt",
            random_state=SEED
        )
        self.X = []
        self.y = []
        self.max_litmus_score = {}
        self.litmus_to_vector_dict = {}
        self.litmus_list = litmus_list
        self.litmus_run_count = defaultdict(int)
        self.max_runs_per_litmus = 10

        # ===== 关键：加载 litmus 向量 =====
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

        # ===== 一致性检查 =====
        for l in self.litmus_to_vector_dict:
            if l not in self.litmus_list:
                raise ValueError(f"Missing vector for litmus: {l}")

        self.logger = get_logger(LOG_NAME)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                name, vec_str = line.split(":", 1)
                vec = eval(vec_str)  # 你这个文件是可信的
                litmus_to_vec[name] = list(vec)

        return litmus_to_vec

    # 加入训练数据
    def add(self, litmus_name, param_vec, score):
        self.logger.info(f"Adding Train {litmus_name} with {param_vec} with score {score}")
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        self.X.append(list(param_vec)+list(litmus_vec))
        self.y.append(score)
        self.max_litmus_score[litmus_name] = max(self.max_litmus_score.get(litmus_name,0), score)

    # 训练 RF
    def fit(self):
        self.model.fit(np.array(self.X), np.array(self.y))

    # EI 计算
    def compute_ei(self, candidate_vecs, litmus_name):
        C = np.array(candidate_vecs)

        preds = np.array([
            est.predict(C) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        eta = self.max_litmus_score.get(litmus_name, -np.inf)

        ei = np.zeros_like(mu)

        # -------- sigma > 0：标准 EI --------
        mask = sigma > 1e-8
        if np.any(mask):
            z = (mu[mask] - eta) / sigma[mask]
            ei[mask] = (
                    (mu[mask] - eta) * norm.cdf(z)
                    + sigma[mask] * norm.pdf(z)
            )

        # -------- sigma == 0：退化为 exploitation --------
        ei[~mask] = np.maximum(mu[~mask] - eta, 0.0)
        # self.logger.info(f"Compute EI {litmus_name} get {ei}")
        return ei

    def compute_ucb(self, candidate_vecs, beta=1.5):
        C = np.array(candidate_vecs)

        preds = np.array([
            est.predict(C) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        return mu + beta * sigma

    def sample_litmus(self, n_litmus):
        """
        随机采样 litmus test（不超过最大运行次数）
        """
        # 1. 过滤掉已经跑满的 litmus
        available = [
            l for l in self.litmus_list
            if self.litmus_run_count[l] < self.max_runs_per_litmus
        ]

        if not available:
            return []

        # 2. 随机采样（不放回）
        k = min(n_litmus, len(available))
        sampled = random.sample(available, k)

        return sampled

    # 候选生成
    def generate_candidates(self, n_litmus=5, n_param=200):
        candidates = []

        litmus_names = self.sample_litmus(n_litmus)

        for litmus in litmus_names:
            litmus_vector = self.litmus_to_vector_dict[litmus]

            for _ in range(n_param):
                p = self.ps.random_vector()
                candidates.append((litmus, list(p)+list(litmus_vector)))


        return candidates


    def groupby_litmus(self, cands):
        """
        cands: List[(litmus_name, full_vec)]
        return: Dict[litmus_name, List[full_vec]]
        """
        groups = defaultdict(list)
        for litmus, vec in cands:
            groups[litmus].append(vec)
        return groups

    def select_batch_next(self, batch_size=8):
        """
        一次性选出 batch_size 个最有希望的候选点
        限制：同一个 batch 内，同一个 litmus test 最多选一次（保证多样性）
        """
        if len(self.X) == 0:
            return []

        # 1. 确保模型是最新的
        self.fit()

        # 2. 生成大量候选点
        # [关键修改]：为了凑够 batch_size 个不重复的 litmus，
        # 我们采样的 litmus 数量必须 > batch_size。这里设为 batch_size * 2 比较稳妥。
        # 如果可用的 litmus 总数少于 batch_size，sample_litmus 内部会自动处理上限。
        n_sample_litmus = max(batch_size * 2, 10)
        cands = self.generate_candidates(n_litmus=n_sample_litmus, n_param=500)

        # 3. 计算所有候选点的 EI
        scored_candidates = []

        # 按 litmus 分组计算 EI
        for litmus, vecs in self.groupby_litmus(cands).items():
            X_arr = np.array(vecs)
            ei_vals = self.compute_ei(X_arr, litmus)

            for i, vec in enumerate(vecs):
                scored_candidates.append({
                    "litmus": litmus,
                    "vec": vec,
                    "ei": ei_vals[i]
                })

        # 4. 全局排序：按 EI 降序排列
        scored_candidates.sort(key=lambda x: x["ei"], reverse=True)

        # 5. [关键修改] 去重选择 Top-K
        best_batch = []
        seen_litmus = set()  # 用于记录本轮已经选过的 litmus

        for item in scored_candidates:
            # 如果凑够了就退出
            if len(best_batch) >= batch_size:
                break

            litmus_name = item["litmus"]

            # 如果这个 litmus 在本轮已经被选过了，直接跳过
            if litmus_name in seen_litmus:
                continue

            # 没选过 -> 加入 batch
            best_batch.append((litmus_name, item["vec"]))
            seen_litmus.add(litmus_name)

        self.logger.info(f"BO Batch Selected {len(best_batch)} unique candidates.")
        return best_batch

    def self_check(self, vec, score):
        X_sample = np.array(vec).reshape(1, -1)
        preds = np.array([
            est.predict(X_sample) for est in self.model.estimators_
        ])

        mu = preds.mean(axis=0)
        self.fit()
        preds = np.array([
            est.predict(X_sample) for est in self.model.estimators_
        ])

        mu_after = preds.mean(axis=0)
        self.logger.info(f"BO self check: {vec}, score: {score}, before: {mu}, after: {mu_after}")


class LitmusRunnerForBayes(LitmusRunner):
    def __init__(
        self,
        litmus_list,
        param_space,
        stat_log,
        mode="time",
        init_samples_per_litmus=3,
        bo_iters=10000,
        pipeline_host="192.168.1.105",
        pipeline_user="root",
        pipeline_pass="riscv",
        pipeline_port=22
    ):
        super().__init__(litmus_list, [], stat_log, mode)

        self.ps = param_space
        self.bo = RandomForestBO(param_space, litmus_list, litmus_vec_path)

        self.init_samples_per_litmus = init_samples_per_litmus
        self.bo_iters = bo_iters

        self.total_runs = 0
        self.init_done = False

        self.cache = ResultCache(stat_log + ".cache.jsonl")
        self.error_cache = ErrorCache(stat_log + ".error.cache.jsonl")
        self.logger = get_logger(LOG_NAME)
        self.cold_init = True

        self.logger.info("Initializing Async Pipeline...")

        self.pipeline = LitmusPipeline(
            host=pipeline_host,
            port=pipeline_port,
            username=pipeline_user,
            password=pipeline_pass,
            remote_work_dir=remote_path # 确保板子上有这个目录
        )
        self.pipeline.start()

    def _submit_one(self, litmus, vec):
        params = self.ps.vector_to_params(vec)
        params.set_riscv_gcc()
        # [关键] 把 vector 挂载到 params 上，方便结果回来时找回
        params._temp_vec = vec

        litmus_file = f"{litmus_path}/{litmus}.litmus"

        self.pipeline.submit_task(
            litmus_path=litmus_file,
            params=params,
            litmus_dir_path=dir_path,  # 本地编译产物路径
            log_dir_path=dir_path,  # 本地结果日志路径
            run_time=1000  # 运行次数/时间
        )

    # === [新增] 辅助函数：解析 Log 算分 ===
    def _parse_log_to_score(self, log_path, litmus_name, has_perple, mode):
        """
        以前 get_score 里既跑又算分，现在 pipeline 跑完了，这里只负责算分。
        """
        try:
            # 这里的逻辑要从你原来的 get_score 或 read_log_to_summary 里提取
            # 示例：
            # res = read_log_to_summary(log_path)
            # return calculate_metric(res)

            # 模拟：假设 file size 代表分数 (请替换为你的真实逻辑)
            # with open(log_path, 'r') as f: ...
            if has_perple:
                return log_path, parse_log_by_mode_perple(log_path, mode)
            else:
                return log_path, parse_log_by_mode(log_path, mode)
        except Exception as e:
            self.logger.error(f"Parse error {log_path}: {e}")
            return -1

    def _preload_and_train(self):
        """
        直接从 self.cache.data 中提取数据进行预热，避免重复读取文件 IO。
        """
        self.logger.info("=== Preloading data from memory (self.cache) for Warm Start ===")

        loaded_count = 0

        # self.cache.data 的格式是: { "litmus_name|0.1,0.2,0.3": score, ... }
        # 我们需要遍历字典，把 key 解析回 litmus 和 param_vec

        for key, score in self.cache.data.items():
            try:
                # 1. 解析 Key
                # 对应 ResultCache._make_key: return f"{litmus}|" + ",".join(map(str, param_vec))
                if "|" not in key:
                    continue

                litmus, param_str = key.split("|", 1)

                # 2. 还原 Param Vector (字符串 -> float列表)
                if not param_str:
                    continue
                param_vec = [float(p) for p in param_str.split(",")]

                # 3. 加入 BO 训练集
                self.bo.add(litmus, param_vec, score)

                # 4. 更新计数 (用于控制采样平衡)
                self.bo.litmus_run_count[litmus] += 1

                loaded_count += 1
            except Exception as e:
                # 容错：防止因历史数据格式差异导致解析失败 crash
                # self.logger.warning(f"Failed to parse cache key: {key}, err: {e}")
                continue

        self.logger.info(f"Loaded {loaded_count} historical records from ResultCache.")

        if loaded_count > 0:
            self.logger.info(f"Training Initial Random Forest Model with {loaded_count} samples...")
            start_t = time.time()
            self.bo.fit()
            self.logger.info(f"Initial Training Complete. Time: {time.time() - start_t:.2f}s")
        else:
            self.logger.warning("Cache is empty in memory! BO will start from scratch (Cold Start).")

    def _initial_sample(self):
        for litmus in self.litmus_list:
            for _ in range(self.init_samples_per_litmus):
                param_vec = self.ps.random_vector()
                # params = self.ps.vector_to_params(param_vec)

                yield litmus, param_vec

    def getNext(self):
        """
        BO-driven task selection
        Return: (litmus_name, params) or None
        """
        # assert False
        # ---------- 终止条件 ----------
        if self.total_runs >= self.bo_iters:
            return None

        # ---------- BO decision ----------
        choice = self.bo.select_batch_next()
        if choice is None:
            return None
        self.cold_init = False
        litmus, full_vec = choice
        param_dim = self.ps.dim
        param_vec = full_vec[:param_dim]

        return litmus, param_vec

        # === [重写] Run 方法 ===
    def run(self):
        self.logger.info("=== Starting Async BO Loop (Warm Start) ===")

        # 配置参数
        TRAIN_TRIGGER = 5  # 每收 5 个结果训练一次
        SUBMIT_SIZE = 8  # 每次生成 8 个新任务
        MAX_QUEUE_SIZE = 20  # 队列最大保持 20 个

        # --- 1. 初始填充 (Pre-fill) ---
        # 因为你有 _preload_and_train，模型已经训练过了 (Warm Start)
        # 所以我们可以直接让 BO 推荐任务，而不需要随机采样！
        self.logger.info("Pre-filling pipeline with BO suggestions...")

        initial_batch = self.bo.select_batch_next(batch_size=20)

        if not initial_batch:
            self.logger.warning("BO returned no suggestions! Fallback to random.")
            # 如果 BO 没吐出东西（极少见），手动随机填几个
            # ... (添加随机填充逻辑) ...
            assert False, "BO returned no suggestions!"
        else:
            for litmus, vec in initial_batch:
                self._submit_one(litmus, vec)

        # --- 2. 流式循环 ---
        pending_results_buffer = []

        # stream_results 是个生成器，会阻塞等待结果
        for result in self.pipeline.stream_results():

            # === A. 处理结果 ===
            task = result['task']
            log_path = result['log_path']
            litmus_name = task.litmus_path.split("/")[-1][:-7]

            # 找回 vector
            if hasattr(task.params, '_temp_vec'):
                param_vec = task.params._temp_vec
            else:
                self.logger.error("Lost vector info in task!")
                continue

            # 算分
            score = self._parse_log_to_score(log_path, litmus_name, task.params.has_perple, self.mode)
            self.logger.info(f"[FINISHED] {litmus_name} Score: {score:.4f}")

            # 存入数据 (Cache + BO)
            self.bo.add(litmus_name, param_vec, score)
            self.cache.add(litmus_name, param_vec, score)
            self.bo.litmus_run_count[litmus_name] += 1
            self.total_runs += 1

            # 加入缓冲区
            pending_results_buffer.append(score)

            # === B. 批次训练 & 更新 (满5个) ===
            if len(pending_results_buffer) >= TRAIN_TRIGGER:
                self.logger.info(f"Collected {len(pending_results_buffer)} results. Training...")

                if self.total_runs < self.bo_iters:
                    try:
                        # 1. 训练模型并生成新任务 (一次生成8个)
                        # 因为是 Warm Start，数据只会越来越多，BO 会越来越准
                        new_batch = self.bo.select_batch_next(batch_size=SUBMIT_SIZE)

                        # 2. 提交新任务
                        for (nxt_litmus, nxt_vec) in new_batch:
                            # 查重：如果 Cache 里已经有了，就没必要再跑了（虽然 select_batch 应该避免）
                            if self.cache.get(nxt_litmus, nxt_vec) is None:
                                self._submit_one(nxt_litmus, nxt_vec)

                        self.logger.info(f"Submitted {len(new_batch)} new tasks.")

                    except Exception as e:
                        self.logger.error(f"BO Step Failed: {e}")

                # 3. [关键] 优胜劣汰：清理 Compile Queue
                # 丢弃积压的旧任务，保证板子跑的都是最新的
                self.pipeline.keep_fresh(max_size=MAX_QUEUE_SIZE)

                # 4. 清空缓冲区
                pending_results_buffer.clear()

            # 退出条件
            if self.total_runs >= self.bo_iters:
                break

        self.logger.info("Run finished. Waiting for pipeline...")
        self.pipeline.wait_completion()
        return self.results

    # def run(self):
    #     """
    #     BO-driven execution loop
    #     """
    #     self.logger.info("=== Starting Over-Provisioning BO Loop ===")
    #     TRAIN_TRIGGER = 5  # 每收回 5 个结果，就训练一次
    #     SUBMIT_SIZE = 8  # 每次产生 8 个新任务 (净赚 +3)
    #     MAX_QUEUE_SIZE = 20  # 队列最长只留 20 个
    #     while True:
    #         nxt = self.getNext()
    #         if nxt is None:
    #             break
    #
    #         litmus, param_vec = nxt
    #         params = self.ps.vector_to_params(param_vec)
    #         params.set_riscv_gcc()
    #         print(f"[RUN] litmus={litmus}, params={params.to_dict()}")
    #         if self.error_cache.has(litmus, param_vec):
    #             self.logger.info(
    #                 f"[SKIP-ERROR] litmus={litmus}, params={param_vec}"
    #             )
    #             continue
    #         # ---------- 查 cache ----------
    #         cached = self.cache.get(litmus, param_vec)
    #         if cached is not None:
    #             score = cached
    #             print(f"[CACHE HIT] litmus={litmus}, score={score:.4f}")
    #         else:
    #             # assert False
    #             print(f"[CHIP RUN] litmus={litmus}, params={params.to_dict()}")
    #             log_path, score = get_score(f"{litmus_path}/{litmus}.litmus", params, mode=self.mode)
    #             if log_path is None:
    #                 self.logger.warning(
    #                     f"[ERROR] litmus={litmus}, params={param_vec}, err=error"
    #                 )
    #                 self.error_cache.add(
    #                     litmus,
    #                     param_vec,
    #                     error_type="runtime_error",
    #                     error_msg="error",
    #                 )
    #                 continue
    #             self.cache.add(litmus, param_vec, score)
    #
    #         self.logger.info(f"Score {litmus} is: {score}")
    #
    #         # 回传给 BO
    #         self.bo.add(litmus, param_vec, score)
    #         self.bo.litmus_run_count[litmus] += 1
    #
    #         self.total_runs += 1
    #         self.results.append((litmus, params, score))
    #
    #         print(
    #             f"[DONE] score={score:.4f}, "
    #             f"best[{litmus}]={self.bo.max_litmus_score[litmus]:.4f}"
    #         )
    #         if not self.cold_init:
    #             self.bo.self_check(param_vec+self.bo.litmus_to_vector_dict[litmus],score)
    #     return self.results




litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
dir_path = "/home/whq/Desktop/code_list/perple_test/bayes_log"
log_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_stat_bayes.csv"
litmus_vec_path="/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector.log"

# litmus_path = "/home/software/桌面/bayes/perple_test_riscv/all_allow_litmus_C910_naive"
# stat_log = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/log_record_bayes.log"
# dir_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_log"
# log_path = "/home/software/桌面/bayes/perple_test_riscv/bayes_stat/log_stat_bayes.csv"
# litmus_vec_path="/home/software/桌面/bayes/perple_test_riscv/bayes_stat/litmus_vector.log"
host = "192.168.226.168"  # 远程服务器地址
# host = "10.42.0.131"
port = 22  # SSH 端口
username = "sipeed"  # SSH 用户名
password = "sipeed"  # SSH 密码
remote_path = "/home/sipeed/test"

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    ts = time.strftime("%Y%m%d-%H%M%S")

    logger = setup_logger(
        log_file=f"{stat_log}.{ts}.run.log",
        level=logging.INFO,
        name=LOG_NAME,
        stdout=True
    )
    logger.info(f"Start BO run | seed={SEED}")

    litmus_list = get_files(litmus_path)
    litmus_list = [path.split("/")[-1][:-7] for path in litmus_list]
    for litmus in litmus_list:
        print(litmus)
    param_space = LitmusParamSpace()
    runner = LitmusRunnerForBayes(
        litmus_list,
        param_space,
        stat_log,
        pipeline_host=host,
        pipeline_user=username,
        pipeline_pass=password,
        pipeline_port=port
    )
    runner._preload_and_train()
    runner.run()
