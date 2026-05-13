# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import datetime

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re) / true_re if abs(true_re - pred_re) / true_re <= 1 else 1


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    return mae, rmse

def save_results_to_txt(params, scores, adjusted_score_list_all, filename="results.txt", adjusted_scores=None,use_xgb= False):
    # 如果文件名中没有时间戳，自动添加
    if "results.txt" in filename and "_" not in filename.replace("results.txt", ""):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{current_time}.txt"

    # 辅助函数：安全计算数值列表的统计量
    def safe_calculate_stats(score_list):
        """安全计算数值列表的均值和标准差"""
        numeric_scores = []
        for score in score_list:
            if score is not None:
                try:
                    # 尝试转换为浮点数
                    numeric_scores.append(float(score))
                except (ValueError, TypeError):
                    # 跳过无法转换的值
                    continue
        if numeric_scores:
            return np.mean(numeric_scores), np.std(numeric_scores)
        else:
            return 0, 0

    with open(filename, 'w') as f:
        # 写入运行时间
        run_time = params.get('run_time', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("=== Battery Prediction Model Results ===\n\n")
        f.write(f"Run Time: {run_time}\n\n")


        if use_xgb:
            # 遍历输出模型参数
            f.write("Model Parameters:\n")
            for key, value in params.items():
                if key != 'xgb_params':
                    f.write(f"{key}: {value}\n")

            if 'xgb_params' in params:
                f.write("\nXGBoost Parameters:\n")
                for key, value in params['xgb_params'].items():
                    f.write(f"  {key}: {value}\n")

            # 检查调整后的结果
            valid_adjusted_scores = [s for s in adjusted_score_list_all if s and len(s) > 0]
            if valid_adjusted_scores:
                f.write("\n=== Attention + XGBoost Adjustment Results ===\n")
                if params['metric'] != 'all':
                    valid_scores = [s[0] for s in valid_adjusted_scores]
                    mean_score, std_score = safe_calculate_stats(valid_scores)
                    f.write(f"{params['metric']} mean: {mean_score:.6f} ± {std_score:.6f}\n")
                else:
                    mlist = ['re', 'mae', 'rmse']
                    for i in range(3):
                        valid_scores = [s[i] for s in valid_adjusted_scores if len(s) > i]
                        mean_score, std_score = safe_calculate_stats(valid_scores)
                        f.write(f"{mlist[i]} mean: {mean_score:.6f} ± {std_score:.6f}\n")
            else:
                f.write("\nNo valid XGBoost adjusted results available")

        else:
            # 遍历输出模型参数
            f.write("Model Parameters:\n")
            for key, value in params.items():
                if key != 'xgb_params':
                    f.write(f"{key}: {value}\n")

            f.write("\n=== Only Attention Model Results ===\n")
            if params['metric'] != 'all':
                valid_scores = [s[0] for s in scores if len(s) > 0]
                mean_score, std_score = safe_calculate_stats(valid_scores)
                f.write(f"{params['metric']} mean: {mean_score:.6f} ± {std_score:.6f}\n")
            else:
                mlist = ['re', 'mae', 'rmse']
                for i in range(3):
                    valid_scores = [s[i] for s in scores if len(s) > i]
                    mean_score, std_score = safe_calculate_stats(valid_scores)
                    f.write(f"{mlist[i]} mean: {mean_score:.6f} ± {std_score:.6f}\n")



def plot_capacity_degradation(Battery, Battery_list, save_path="capacity_degradation.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c.']

    for name, color in zip(Battery_list, color_list):
        df_result = Battery[name]
        ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_' + name)

    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
           title='Capacity degradation at ambient temperature of 1°C')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

