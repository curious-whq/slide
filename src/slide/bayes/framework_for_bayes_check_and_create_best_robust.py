import json
import logging
import os
import random
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from src.slide.bayes.litmus_param_space import LitmusParamSpace
from src.slide.bayes.logger_util import setup_logger, get_logger
from src.slide.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_final_production"
ALPHA = 1.0


class RandomForestBO:
    def __init__(self, param_space: LitmusParamSpace, litmus_names, n_estimators=200, litmus_vec_path=""):
        self.ps = param_space

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
            random_state=SEED
        )
        self.X = []
        self.y = []

        # 加载特征辅助数据
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)
        self.litmus_id_map = {name: i for i, name in enumerate(litmus_names)}

        # === 关键修复：自动计算向量维度 ===
        if self.litmus_to_vector_dict:
            # 取第一个向量的长度作为标准长度
            self.vec_dim = len(next(iter(self.litmus_to_vector_dict.values())))
        else:
            self.vec_dim = 12  # 如果完全没加载到文件，才使用默认值

        print(f"Detected Litmus Vector Dimension: {self.vec_dim}")

        self.is_fitted = False

    def load_litmus_vectors(self, path):
        d = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    if ":" in line:
                        try:
                            n, v = line.strip().split(":", 1)
                            d[n] = eval(v)
                        except:
                            pass
        return d

    def _get_features(self, litmus_name, param_vec):
        # 1. Params (11)
        feats = list(param_vec)

        # 2. Vector (使用动态维度)
        if litmus_name in self.litmus_to_vector_dict:
            feats.extend(self.litmus_to_vector_dict[litmus_name])
        else:
            # 补0时使用检测到的维度，而不是写死 12
            feats.extend([0] * self.vec_dim)

        # 3. ID (1)
        lid = self.litmus_id_map.get(litmus_name, -1)
        feats.append(lid)

        return feats

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_id_map: return

        feat = self._get_features(litmus_name, param_vec)
        self.X.append(feat)
        self.y.append(score)

    def fit(self):
        if not self.X:
            print("WARNING: No data to fit!")
            return

        X_arr = np.array(self.X)
        y_arr = np.array(self.y)

        # 再次确认维度一致性 (调试用)
        if X_arr.shape[1] != (11 + self.vec_dim + 1):
            print(f"WARNING: Feature dimension mismatch! Expected {11 + self.vec_dim + 1}, got {X_arr.shape[1]}")

        self.y_train_log = np.log1p(y_arr)

        self.model.fit(X_arr, self.y_train_log)
        self.is_fitted = True

    def predict_challenger(self, litmus_name, candidate_params_matrix, best_history_score_real):
        if not self.is_fitted:
            return 0, 0.0, 0.0, False

        n_candidates = len(candidate_params_matrix)

        # 1. 构造矩阵
        X_base = candidate_params_matrix

        # Vector
        if litmus_name in self.litmus_to_vector_dict:
            vec = self.litmus_to_vector_dict[litmus_name]
        else:
            # === 关键修复：使用 self.vec_dim ===
            vec = [0] * self.vec_dim

        X_vec = np.tile(vec, (n_candidates, 1))

        # ID
        lid = self.litmus_id_map.get(litmus_name, -1)
        X_id = np.full((n_candidates, 1), lid)

        # Concat
        X_batch = np.hstack([X_base, X_vec, X_id])

        # 2. 预测
        all_preds = []
        for tree in self.model.estimators_:
            all_preds.append(tree.predict(X_batch))
        all_preds = np.array(all_preds)

        # 3. Log 空间统计
        mu_log = np.mean(all_preds, axis=0)
        sigma_log = np.std(all_preds, axis=0)

        # 4. LCB 决策
        lcb_log = mu_log - (ALPHA * sigma_log)

        best_idx = np.argmax(lcb_log)
        best_lcb_val = lcb_log[best_idx]

        # 5. 挑战判定
        hist_best_log = np.log1p(best_history_score_real)
        is_stronger = best_lcb_val > (hist_best_log + 1e-4)

        pred_mean_real = np.expm1(mu_log[best_idx])

        return best_idx, pred_mean_real, sigma_log[best_idx], is_stronger


if __name__ == "__main__":
    # 配置
    litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
    stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
    # 使用包含数据的 cache 文件
    cache_file_path = stat_log_base + ".cache_sum_70_no_norm_for_graph.jsonl"
    litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_dnn_gt0.log"

    logger = setup_logger(f"{stat_log_base}.final.run.log", logging.INFO, LOG_NAME, True)
    logger.info(f"=== Generating FINAL One-Shot Recommendations | Alpha={ALPHA} ===")

    # 1. 准备 ID Map
    full_litmus_list = get_files(litmus_path)
    # 去掉后缀
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    param_space = LitmusParamSpace()
    bo = RandomForestBO(param_space, litmus_names, n_estimators=200, litmus_vec_path=litmus_vec_path)

    # 2. 加载历史数据
    logger.info(f"Loading history from {cache_file_path} ...")
    incumbents = {}
    valid_cnt = 0

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        lname = obj["litmus"]

                        # 过滤无效ID
                        if lname not in bo.litmus_id_map: continue

                        vec = obj["param"]
                        sc = obj["score"]

                        bo.add(lname, vec, sc)
                        valid_cnt += 1

                        if lname not in incumbents:
                            incumbents[lname] = {'param': vec, 'score': sc}
                        else:
                            if sc > incumbents[lname]['score']:
                                incumbents[lname] = {'param': vec, 'score': sc}
                    except:
                        pass
    else:
        logger.error("Cache file not found!")
        exit(1)

    logger.info(f"Loaded {valid_cnt} valid records.")

    # 3. 训练
    logger.info("Training Model...")
    bo.fit()

    # 4. 生成候选集
    all_candidates = param_space.get_all_combinations()
    X_base_params = np.array(all_candidates)

    # 5. 决策循环
    final_decisions = {}
    stats = {"HIST": 0, "NEW": 0}

    logger.info("Making decisions for all files...")
    total = len(litmus_names)
    count = 0

    for litmus in litmus_names:
        count += 1
        if count % 100 == 0: logger.info(f"Processing {count}/{total}...")

        has_history = litmus in incumbents
        if has_history:
            hist_score = incumbents[litmus]['score']
            hist_param = incumbents[litmus]['param']
        else:
            hist_score = -1.0
            hist_param = None

        best_idx, pred_mean, pred_sigma, is_stronger = bo.predict_challenger(
            litmus, X_base_params, hist_score
        )

        # 决策逻辑
        if not has_history:
            final_param = all_candidates[best_idx]
            source = "NEW (No Hist)"
            stats["NEW"] += 1
        elif is_stronger:
            final_param = all_candidates[best_idx]
            source = "NEW (Better)"
            stats["NEW"] += 1
        else:
            final_param = hist_param
            source = "HIST"
            stats["HIST"] += 1

        final_decisions[litmus] = {
            "param": final_param,
            "source": source,
            "pred_score": float(pred_mean)
        }

    out_file = "best_params_final.json"
    with open(out_file, "w") as f:
        json.dump(final_decisions, f, indent=4)

    logger.info("=" * 60)
    logger.info(f"Saved to {out_file}")
    logger.info(f"Stats: {stats}")