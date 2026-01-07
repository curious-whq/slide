import json
import logging
import os
import random
import time
from collections import defaultdict

# === 核心修改：引入 XGBoost 和 Numpy ===
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from src.slide.bayes.litmus_param_space import LitmusParamSpace
from src.slide.bayes.logger_util import setup_logger, get_logger
from src.slide.bayes.util import get_files

SEED = 2025
LOG_NAME = "bayes_eval"


# ================= 类定义 =================

class XGBoostEnsembleBO:
    """
    使用 Bagging 集成的 XGBoost 贝叶斯优化器。
    目的：
    1. 利用 XGBoost 提升预测精度 (Step 1)。
    2. 利用多模型集成的方差来评估不确定性 (Step 2)。
    """

    def __init__(self, param_space: LitmusParamSpace, litmus_list, n_estimators=100, n_models=5,
                 litmus_vec_path=""):
        self.ps = param_space
        self.n_models = n_models  # 集成模型的数量，建议 5-10 个
        self.models = []
        self.litmus_list = litmus_list
        self.logger = get_logger(LOG_NAME)

        # 初始化多个 XGBoost 模型，赋予不同的随机种子
        for i in range(self.n_models):
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=5,  # 树深，稍微控制防止过拟合
                learning_rate=0.06,  # 学习率
                subsample=0.8,  # 样本采样，制造差异性
                colsample_bytree=0.8,  # 特征采样
                n_jobs=-1,
                random_state=SEED + i,  # === 关键：每个模型种子不同 ===
                objective='reg:squarederror'
            )
            self.models.append(model)

        self.X = []
        self.y = []
        self.litmus_to_vector_dict = self.load_litmus_vectors(litmus_vec_path)

    def load_litmus_vectors(self, path):
        litmus_to_vec = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line: continue
                    try:
                        name, vec_str = line.split(":", 1)
                        vec = eval(vec_str)
                        litmus_to_vec[name] = list(vec)
                    except:
                        pass
        return litmus_to_vec

    def add(self, litmus_name, param_vec, score):
        if litmus_name not in self.litmus_to_vector_dict:
            return
        litmus_vec = self.litmus_to_vector_dict[litmus_name]
        # 简单的特征拼接
        feature = list(param_vec) + list(litmus_vec)
        self.X.append(feature)
        self.y.append(score)

    def fit(self):
        self.logger.info(f"Start fitting {self.n_models} XGBoost models...")
        if not self.X:
            return

        X_arr = np.array(self.X)
        # Log1p 处理：压缩目标值范围，让模型更容易学
        y_arr = np.log1p(np.array(self.y))

        # 训练所有子模型
        for idx, model in enumerate(self.models):
            model.fit(X_arr, y_arr)
            # self.logger.info(f"Sub-model {idx+1}/{self.n_models} trained.")

    def predict_batch_with_uncertainty(self, litmus_list, param_list):
        """
        批量预测，并返回 (Mean, Std)
        """
        X_batch = []
        valid_indices = []

        # 1. 构建特征
        for i, (litmus, param) in enumerate(zip(litmus_list, param_list)):
            if litmus in self.litmus_to_vector_dict:
                litmus_vec = self.litmus_to_vector_dict[litmus]
                X_batch.append(list(param) + list(litmus_vec))
                valid_indices.append(i)

        if not X_batch:
            return [], [], []

        X_batch_np = np.array(X_batch)

        # 2. 所有子模型分别预测
        # shape: (n_models, n_samples)
        all_preds_log = np.zeros((self.n_models, len(X_batch)))

        for idx, model in enumerate(self.models):
            all_preds_log[idx, :] = model.predict(X_batch_np)

        # 3. 将预测值还原回线性空间 (expm1)
        # 建议先还原再算均值方差，这样方差具有物理意义
        all_preds_linear = np.expm1(all_preds_log)

        # 4. 计算均值和标准差
        mean_preds = np.mean(all_preds_linear, axis=0)
        std_preds = np.std(all_preds_linear, axis=0)

        return mean_preds, std_preds, valid_indices


# ================= 主程序部分修改 =================

# 配置路径 (保持你的路径不变)
litmus_path = "/home/whq/Desktop/code_list/perple_test/all_allow_litmus_C910_naive"
stat_log_base = "/home/whq/Desktop/code_list/perple_test/bayes_stat/log_record_bayes.log"
litmus_vec_path = "/home/whq/Desktop/code_list/perple_test/bayes_stat/litmus_vector4_two_tower_gt0.log"
cache_file_path = stat_log_base + ".cache_sum_70_no_norm_gt_0_for_graph.jsonl"

if __name__ == "__main__":
    # 1. Setup
    random.seed(SEED)
    np.random.seed(SEED)
    log_file_name = f"{stat_log_base}.xgboost_check.log"
    logger = setup_logger(log_file=log_file_name, level=logging.INFO, name=LOG_NAME, stdout=True)
    logger.info(f"=== Start Evaluation (XGBoost Ensemble) | Seed={SEED} ===")

    # 2. Load Litmus
    full_litmus_list = get_files(litmus_path)
    litmus_names = [path.split("/")[-1][:-7] for path in full_litmus_list]

    # 3. 初始化 BO (使用新的 Ensemble 类)
    param_space = LitmusParamSpace()
    # n_models=5 表示训练5个XGBoost来投票，n_estimators可以稍微小一点比如100
    bo = XGBoostEnsembleBO(
        param_space,
        litmus_names,
        n_estimators=100,
        n_models=5,
        litmus_vec_path=litmus_vec_path
    )

    # 4. Load Data
    all_data = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass

    # 5. Split Data
    # 既然数据少，为了验证效果，这里可以稍微多留一点训练集，或者维持 7:3
    random.shuffle(all_data)
    train_size = int(len(all_data) * 0.7)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    logger.info(f"Train size: {len(train_data)} | Test size: {len(test_data)}")

    # 6. Fit
    for item in train_data:
        bo.add(item["litmus"], item["param"], item["score"])
    bo.fit()

    # 7. Batch Predict with Uncertainty
    logger.info("Evaluating on test set...")
    test_litmus = [item["litmus"] for item in test_data]
    test_params = [item["param"] for item in test_data]

    t0 = time.time()
    # === 调用带不确定性的预测 ===
    means, stds, valid_indices = bo.predict_batch_with_uncertainty(test_litmus, test_params)
    logger.info(f"Prediction finished in {time.time() - t0:.4f}s")

    # 8. 结果重组与分析
    groups = defaultdict(list)
    y_true_all = []
    y_pred_all = []

    for ptr, original_idx in enumerate(valid_indices):
        item = test_data[original_idx]
        actual = item["score"]
        pred_mean = means[ptr]
        pred_std = stds[ptr]  # 获取不确定性

        groups[item["litmus"]].append({
            'param': item["param"],
            'actual': actual,
            'pred': pred_mean,
            'std': pred_std  # 记录不确定性
        })
        y_true_all.append(actual)
        y_pred_all.append(pred_mean)

    # 9. 统计展示
    logger.info("=" * 80)
    logger.info(f"{'LITMUS':<25} | {'TOP1?':<5} | {'ACTUAL':<6} | {'PRED':<6} | {'STD(Uncertainty)':<15}")
    logger.info("-" * 80)

    total_cnt = 0
    top1_cnt = 0
    top3_cnt = 0

    for litmus, records in groups.items():
        if len(records) < 2: continue
        total_cnt += 1

        # 真实最优
        records.sort(key=lambda x: x['actual'], reverse=True)
        best_actual = records[0]['actual']

        # 模型预测最优 (目前还是按 Mean 排序，后续你可以改成 UCB: Mean + 1.96*Std)
        records.sort(key=lambda x: x['pred'], reverse=True)
        best_pred_record = records[0]

        # Top-1 Hit
        if best_pred_record['actual'] >= best_actual:
            top1_cnt += 1
            hit = "YES"
        else:
            hit = "NO"

        # Top-3 Hit
        top3_preds = records[:3]
        if any(r['actual'] >= best_actual for r in top3_preds):
            top3_cnt += 1

        # 仅打印一部分日志
        if total_cnt <= 20 or hit == "NO":  # 打印前20个，或者打印预测错误的
            logger.info(
                f"{litmus[:25]:<25} | {hit:<5} | {best_actual:<6.2f} | {best_pred_record['pred']:<6.2f} | {best_pred_record['std']:<15.4f}")

    # 10. 全局指标
    if total_cnt > 0:
        logger.info("=" * 60)
        logger.info(f"Top-1 Accuracy: {top1_cnt / total_cnt * 100:.2f}%")
        logger.info(f"Top-3 Recall:   {top3_cnt / total_cnt * 100:.2f}%")

        rho, _ = spearmanr(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)

        logger.info(f"Global Spearman Rho: {rho:.4f}")
        logger.info(f"Global R^2:          {r2:.4f}")
        logger.info(f"Global MAE:          {mae:.4f}")
        logger.info("=" * 60)