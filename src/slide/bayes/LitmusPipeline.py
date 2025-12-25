import os
import time
import uuid
import threading
import queue
import paramiko
from dataclasses import dataclass

from src.slide.utils.cmd_util import run_cmd


# 假设 params 和 run_cmd 等外部依赖已经定义
# from your_module import LitmusParams, run_cmd, ...

@dataclass
class LitmusTask:
    litmus_path: str
    params: object  # LitmusParams
    litmus_dir_path: str
    log_dir_path: str
    run_time: int

    # 运行时生成的属性
    unique_id: str = None
    local_exe_path: str = None
    remote_exe_path: str = None
    remote_log_path: str = None
    local_log_path: str = None


class LitmusPipeline:
    def __init__(self, host, port, username, password, remote_work_dir="/tmp/litmus_work"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_work_dir = remote_work_dir

        # 建立 SSH 连接池 (为了并发，建议 SFTP 和 Exec 分开)
        self.ssh_client = self._connect_ssh()
        self.sftp_client = self.ssh_client.open_sftp()

        # 定义四个队列：编译等待、上传等待、运行等待、下载等待
        self.compile_queue = queue.Queue()
        self.upload_queue = queue.Queue()
        self.run_queue = queue.Queue()
        self.download_queue = queue.Queue()

        # 结果列表
        self.result_queue = queue.Queue()
        self.running = True

    def stream_results(self):
        """
        这是一个生成器。
        主线程调用它时，就像在遍历一个永远读不完的列表。
        """
        while self.running:  # 只要流水线没关，就一直循环
            try:
                # 1. 尝试从结果队列里拿东西
                # timeout=1 表示：如果队列空了，我只等 1 秒。
                # 如果 1 秒后还是空的，抛出 Empty 异常，让我有机会检查 self.running 还在不在。
                result = self.result_queue.get(timeout=1)

                # 2. 拿到结果了！把结果“吐”给主线程
                # 此时，stream_results 函数会在这里“暂停”！
                # 直到主线程处理完这个 result，进入下一次循环，这里才会“恢复”继续运行。
                yield result

            except queue.Empty:
                # 队列是空的，没事，继续下一轮循环，看看 running 是不是变成 False 了
                # 如果所有任务都做完了，我们可以加入退出逻辑
                if self.compile_queue.empty() and \
                        self.upload_queue.empty() and \
                        self.run_queue.empty() and \
                        self.download_queue.empty() and \
                        self.result_queue.empty():
                    # 再次确认真的没任务了，且 worker 都不忙了（这里简化判断）
                    # 在你的逻辑里，通常由主线程决定 break，所以这里 continue 就行
                    continue
                continue


    def keep_fresh(self, max_size=20):
        """
        [修正版] 源头截流：清理待编译队列。
        如果待编译的任务太多，说明后端堵死了，直接把积压在编译队列里的旧任务扔掉。
        这样既省了编译 CPU，又省了上传带宽。
        """
        # 1. 检查 compile_queue (源头)
        q_size = self.compile_queue.qsize()

        # 我们希望保留一定的 buffer，但不要太多
        # 注意：这里我们设定 compile_queue 的阈值。
        # 如果 Runner 很慢，Compile Queue 会迅速堆积。
        if q_size > max_size:
            discard_count = q_size - max_size
            print(f"[Pipeline] System congested. Discarding {discard_count} stale tasks from COMPILE queue...")

            for _ in range(discard_count):
                try:
                    # get() 拿出来的是最老的任务（FIFO）
                    task = self.compile_queue.get_nowait()

                    # 直接丢弃，不做任何处理
                    # 也不需要删文件，因为文件根本还没生成！
                    print(f"   -> Dropped task {task.unique_id} (Saved compile time)")

                    self.compile_queue.task_done()
                except queue.Empty:
                    break

    def _connect_ssh(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, self.port, self.username, self.password)
        # 预先创建远程工作目录
        ssh.exec_command(f"mkdir -p {self.remote_work_dir}")
        return ssh

    def submit_task(self, litmus_path, params, litmus_dir_path, log_dir_path, run_time=100):
        """提交任务入口"""
        task = LitmusTask(litmus_path, params, litmus_dir_path, log_dir_path, run_time)
        task.unique_id = str(uuid.uuid4())[:8]  # 生成短UUID防止文件名冲突
        self.compile_queue.put(task)

    # --- Worker 1: 编译器 (本地 CPU 密集型) ---
    def worker_compiler(self):
        # 策略：根据你的机器核心数动态调整。
        # 如果你开 4 个编译线程，每个 make 用 4 核，总共占用 16 核。
        # 这里的 -j4 是指单个 make 内部允许并行的任务数。
        make_jobs = 2

        while self.running:
            try:
                task = self.compile_queue.get(timeout=2)
            except queue.Empty:
                continue

            print(f"[Compiler] Building {task.litmus_path}...")

            litmus_name = task.litmus_path.split("/")[-1][:-7]
            litmus_dir = os.path.join(task.litmus_dir_path, f"{litmus_name}_{str(task.params)}")
            exe_path = os.path.join(litmus_dir, "run.exe")

            # 模拟检查或生成逻辑
            if not os.path.exists(exe_path):
                # 1. 生成代码 (假设这是单线程极快操作)
                run_cmd(task.params.to_litmus7_format(task.litmus_path, litmus_dir))

                # 2. 核心修改点：使用 -j 参数并行编译
                # 这里的 run_cmd 需要是你封装好的能够执行 shell 的函数
                # 加上 > /dev/null 减少控制台 IO 输出，提高速度
                build_cmd = f"cd {litmus_dir}; make -j{make_jobs} > /dev/null 2>&1"
                os.system(build_cmd)

                # 设置路径
            task.local_exe_path = exe_path
            task.remote_exe_path = os.path.join(self.remote_work_dir, f"run_{task.unique_id}.exe")
            task.remote_log_path = os.path.join(self.remote_work_dir, f"run_{task.unique_id}.log")

            self.upload_queue.put(task)
            self.compile_queue.task_done()

    # --- Worker 2: 上传器 (IO 密集型) ---
    def _reconnect(self):
        """内部辅助函数：强制重连 SSH 和 SFTP"""
        print("[Uploader] Detected broken connection. Reconnecting...")
        try:
            # 1. 先尝试关闭旧连接（忽略错误）
            if self.sftp_client: self.sftp_client.close()
            if self.ssh_client: self.ssh_client.close()
        except Exception:
            pass

        # 2. 重新建立连接 (把你的连接逻辑复制到这里，或者调用你类里已有的 connect 方法)
        # 假设你有一个 setup_ssh() 或者类似的初始化逻辑
        # self.setup_ssh()
        # 下面是手动重连的示例：
        import paramiko
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=self.host, username=self.username, password=self.password)  # 使用你的参数
        self.sftp_client = self.ssh_client.open_sftp()
        print("[Uploader] Reconnection successful.")

    def worker_uploader(self):
        while self.running:
            try:
                task = self.upload_queue.get(timeout=2)
            except queue.Empty:
                continue

            print(f"[Uploader] Uploading {task.local_exe_path} -> {task.remote_exe_path}")

            # === 增加重试循环 ===
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 1. 检查连接是否还活着 (可选，但推荐)
                    if self.ssh_client.get_transport() is None or not self.ssh_client.get_transport().is_active():
                        raise Exception("Transport is not active")

                    # 2. 执行上传和命令
                    self.sftp_client.put(task.local_exe_path, task.remote_exe_path)
                    self.ssh_client.exec_command(f"chmod +x {task.remote_exe_path}")

                    # 3. 成功后放入下一级队列并退出重试循环
                    self.run_queue.put(task)
                    break

                except Exception as e:
                    print(f"[Error] Upload failed (Attempt {attempt + 1}/{max_retries}): {e}")

                    # 如果是最后一次尝试依然失败，则彻底放弃该任务（防止死循环）
                    if attempt == max_retries - 1:
                        print(f"[Error] Critical: Failed to upload {task.litmus_path} after retries.")
                    else:
                        # 否则，尝试重连，然后进行下一次循环
                        self._reconnect()

            self.upload_queue.task_done()

    # --- Worker 3: 执行器 (远程瓶颈资源) ---
    def worker_runner(self):
        """
        这个线程必须保证：只要队列里有，就立刻发命令，绝不休息。
        因为上传是独立的，所以当我们跑 Task N 时，Task N+1 正在后台上传。
        """
        while self.running:
            try:
                task = self.run_queue.get(timeout=2)
            except queue.Empty:
                continue

            print(f"[Runner] >>> Running {task.unique_id} on remote chip...")

            # 执行命令（阻塞等待远程结束）
            cmd = f"{task.remote_exe_path} -s {task.run_time} > {task.remote_log_path} 2>&1"
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

            exit_status = stdout.channel.recv_exit_status()  # 这一步是真正的耗时步

            if exit_status == 0:
                print(f"[Runner] <<< Finished {task.unique_id}")
            else:
                print(f"[Runner] Failed {task.unique_id} status {exit_status}")

            self.download_queue.put(task)
            self.run_queue.task_done()

    # --- Worker 4: 下载器 (修改版：支持多线程，独立SFTP通道) ---
    def worker_downloader(self):
        # [关键] 每个线程创建自己独立的 sftp 客户端，互不抢占通道
        # 既然 ssh_client 已经建立了，open_sftp() 只是在现有连接上开一个新通道，开销很小
        thread_sftp = self.ssh_client.open_sftp()

        while self.running:
            try:
                task = self.download_queue.get(timeout=2)
            except queue.Empty:
                continue

            litmus_name = task.litmus_path.split("/")[-1][:-7]
            # 构造本地文件名
            task.local_log_path = os.path.join(
                task.log_dir_path,
                f"{litmus_name}_{str(task.params)}_{task.run_time}-{task.unique_id}.log"
            )

            # print(f"[Downloader] pulling {task.remote_log_path}...")
            try:
                # 使用该线程独享的 sftp 下载
                thread_sftp.get(task.remote_log_path, task.local_log_path)

                # 下载完顺手删掉远程文件 (Log 和 Exe 都删掉)
                # 这一步也可以用 sftp.remove，比 exec_command 快，因为不需要开新 shell
                try:
                    thread_sftp.remove(task.remote_log_path)
                    thread_sftp.remove(task.remote_exe_path)
                except Exception:
                    pass  # 文件不在就算了

                # 放入结果队列
                result_package = {'task': task, 'log_path': task.local_log_path}
                self.result_queue.put(result_package)

            except Exception as e:
                print(f"[Error] Download failed: {e}")

            self.download_queue.task_done()

        # 线程退出前关闭 sftp
        thread_sftp.close()

    def start(self, compiler_thread_count=4, downloader_thread_count=2):  # <--- 新增参数
        threads = []

        print(f"Starting {compiler_thread_count} compiler threads...")
        for _ in range(compiler_thread_count):
            t = threading.Thread(target=self.worker_compiler, daemon=True)
            t.start()
            threads.append(t)

        # Uploader (可以开 2 个)
        for _ in range(2):
            t = threading.Thread(target=self.worker_uploader, daemon=True)
            t.start()
            threads.append(t)

        # Runner (必须 1 个)
        t = threading.Thread(target=self.worker_runner, daemon=True)
        t.start()
        threads.append(t)

        # Downloader (动态配置)
        print(f"Starting {downloader_thread_count} downloader threads...")
        for _ in range(downloader_thread_count):
            t = threading.Thread(target=self.worker_downloader, daemon=True)
            t.start()
            threads.append(t)

        return threads

    def wait_completion(self):
        # 等待所有队列清空
        self.compile_queue.join()
        self.upload_queue.join()
        self.run_queue.join()
        self.download_queue.join()
        self.running = False
        print("All tasks completed.")


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化 Pipeline
    pipeline = LitmusPipeline(
        host="192.168.1.x", port=22,
        username="user", password="pwd",
        remote_work_dir="/home/user/litmus_ci"
    )
    pipeline.start()

    # 2. 疯狂塞入任务
    # 假设你有一个任务列表 list_of_litmus_files
    # for l_path in list_of_litmus_files:
    #     pipeline.submit_task(l_path, some_params, local_dir, log_dir)

    # 3. 等待结束
    pipeline.wait_completion()