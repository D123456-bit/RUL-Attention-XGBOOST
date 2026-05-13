# import numpy as np
# from config import *
# from data_loader import load_battery_data
# from model import AttModel
# from trainer_xgboots import train_model
# from utils import save_results_to_txt, plot_capacity_degradation
# import datetime
#
# def main():
#     print(f"Using device: {device}")
#     # 获取当前时间戳
#     current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     print(f"Run started at: {current_time}")
#
#     # 加载数据
#     Battery = load_battery_data(Battery_list, dir_path)
#
#     # 绘制容量衰减图
#     plot_capacity_degradation(Battery, Battery_list)
#
#     # 模型参数
#     model_params = {
#         'feature_size': feature_size,
#         'hidden_dim': hidden_dim,
#         'num_layers': num_layers,
#         'nhead': nhead,
#         'dropout_att': dropout_att,
#         'device': device,
#
#     }
#
#     # 训练参数
#     train_params = {
#         'lr': lr,
#         'weight_decay': weight_decay,
#         'EPOCH': EPOCH,
#         'seeds': seeds,
#         'metric': metric,
#         'model_class': AttModel,
#         'run_time': current_time  # 保存运行时间到参数中
#     }
#
#
#
#     if use_xgb:
#         # 使用注意力模型 + XGBoost调整
#         score_list, result_list, adjusted_score_list_all = train_model(
#             Battery, Battery_list, type('Config', (), {
#                 'device': device,
#                 'Rated_Capacity': Rated_Capacity,
#                 'window_size': window_size
#             }), model_params,
#             {**train_params, 'model_type': 'xgb'},  # 设置模型类型
#             xgb_params
#         )
#
#     else:
#         # 仅使用注意力模型
#         score_list, result_list, adjusted_score_list_all = train_model(
#             Battery, Battery_list, type('Config', (), {
#                 'device': device,
#                 'Rated_Capacity': Rated_Capacity,
#                 'window_size': window_size
#             }), model_params, train_params
#         )
#
#     # # 训练模型
#     # score_list, result_list, adjusted_score_list_all = train_model(
#     #     Battery, Battery_list,
#     #     type('Config', (), {
#     #         'device': device,
#     #         'Rated_Capacity': Rated_Capacity,
#     #         'window_size': window_size
#     #     }),
#     #     model_params,
#     #     train_params,
#     #     xgb_params
#     # )
#
#     # 保存结果
#     params_to_save = {
#         'feature_size': feature_size,
#         'hidden_dim': hidden_dim,
#         'num_layers': num_layers,
#         'nhead': nhead,
#         'dropout_att': dropout_att,
#         'lr': lr,
#         'weight_decay': weight_decay,
#         'EPOCH': EPOCH,
#         'metric': metric,
#         'seeds': seeds,
#         'Rated_Capacity': Rated_Capacity,
#         # 'model_type': 'Only_Attention_with_XGBoost_Adjustment',
#         'xgb_params': xgb_params
#     }
#
#     # 保存原始和调整后的结果
#
#
#
#     # 打印结果 - 修复索引错误
#     print("\n=== Final Results ===")
#     if use_xgb:
#         results_filename = f"attention_xgb_results_{current_time}.txt"
#         save_results_to_txt(params_to_save, score_list, adjusted_score_list_all, results_filename, use_xgb=use_xgb)
#         # 检查调整后的结果是否有效
#         valid_adjusted_scores = [s for s in adjusted_score_list_all if s and len(s) > 0]
#         if valid_adjusted_scores:
#             print("\nAfter XGBoost Adjustment:")
#             if metric != 'all':
#                 valid_scores = [s[0] for s in valid_adjusted_scores]
#                 mean_score = np.mean(valid_scores) if valid_scores else 0
#                 print(f'{metric} mean: {mean_score:.6f}')
#             else:
#                 mlist = ['re', 'mae', 'rmse']
#                 for i in range(3):
#                     valid_scores = [s[i] for s in valid_adjusted_scores if len(s) > i]
#                     mean_score = np.mean(valid_scores) if valid_scores else 0
#                     print(f'{mlist[i]} mean: {mean_score:.6f}')
#         else:
#             print("\nNo valid XGBoost adjusted results available")
#     else:
#         results_filename = f"attention_results_{current_time}.txt"
#         save_results_to_txt(params_to_save, score_list, adjusted_score_list_all, results_filename, use_xgb=use_xgb)
#         print("Only Attention Model:")
#         if metric != 'all':
#             valid_scores = [s[0] for s in score_list if len(s) > 0]
#             mean_score = np.mean(valid_scores) if valid_scores else 0
#             print(f'{metric} mean: {mean_score:.6f}')
#         else:
#             mlist = ['re', 'mae', 'rmse']
#             for i in range(3):
#                 valid_scores = [s[i] for s in score_list if len(s) > i]
#                 mean_score = np.mean(valid_scores) if valid_scores else 0
#                 print(f'{mlist[i]} mean: {mean_score:.6f}')
#
#
# if __name__ == "__main__":
#     main()

import numpy as np
from config import *
from data_loader import load_battery_data
from model import AttModel
from trainer_xgboots import train_model, plot_predictions_vs_actual
from utils import save_results_to_txt, plot_capacity_degradation
import datetime
import os

def main():
    print(f"Using device: {device}")
    # 获取当前时间戳
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run started at: {current_time}")

    # 创建结果目录
    results_dir = f"results_{current_time}"
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    Battery = load_battery_data(Battery_list, dir_path)

    # 绘制容量衰减图
    plot_capacity_degradation(Battery, Battery_list)

    # 模型参数
    model_params = {
        'feature_size': feature_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'nhead': nhead,
        'dropout_att': dropout_att,
        'device': device,

    }

    # 训练参数
    train_params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'EPOCH': EPOCH,
        'seeds': seeds,
        'metric': metric,
        'model_class': AttModel,
        'run_time': current_time  # 保存运行时间到参数中
    }

    # 训练模型并获取测试数据
    if use_xgb:
        score_list, result_list, adjusted_score_list_all, test_data_list = train_model(
            Battery, Battery_list, type('Config', (), {
                'device': device,
                'Rated_Capacity': Rated_Capacity,
                'window_size': window_size
            }), model_params,
            {**train_params, 'model_type': 'xgb'},
            xgb_params
        )
    else:
        score_list, result_list, adjusted_score_list_all, test_data_list = train_model(
            Battery, Battery_list, type('Config', (), {
                'device': device,
                'Rated_Capacity': Rated_Capacity,
                'window_size': window_size
            }), model_params, train_params
        )

    # 绘制每个电池的预测 vs 实际图
    print("\n=== Generating Prediction vs Actual Plots ===")

    # 计算每个电池有多少个种子运行
    runs_per_battery = len(seeds)

    for battery_idx, battery_name in enumerate(Battery_list):
        for run_idx in range(runs_per_battery):
            # 计算在列表中的索引
            list_idx = battery_idx * runs_per_battery + run_idx

            if list_idx < len(test_data_list) and list_idx < len(result_list):
                test_data = test_data_list[list_idx]
                predictions = result_list[list_idx]

                if len(test_data) == len(predictions):
                    # 绘制图形
                    plot_filename = f"{battery_name}_seed{seeds[run_idx]}_pred_vs_actual.png"
                    plot_path = os.path.join(results_dir, plot_filename)

                    plot_predictions_vs_actual(
                        test_data=test_data,
                        predictions=predictions,
                        battery_name=f"{battery_name} (Seed {seeds[run_idx]})",
                        save_path=plot_path
                    )

                    print(f"Generated plot for {battery_name} with seed {seeds[run_idx]}")
                else:
                    print(f"Data length mismatch for {battery_name} seed {seeds[run_idx]}: "
                          f"test_data={len(test_data)}, predictions={len(predictions)}")

    # 保存结果到文件
    params_to_save = {
        'feature_size': feature_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'nhead': nhead,
        'dropout_att': dropout_att,
        'lr': lr,
        'weight_decay': weight_decay,
        'EPOCH': EPOCH,
        'metric': metric,
        'seeds': seeds,
        'Rated_Capacity': Rated_Capacity,
        'xgb_params': xgb_params if use_xgb else None
    }

    # 保存结果
    results_filename = os.path.join(results_dir, f"results_{current_time}.txt")
    save_results_to_txt(params_to_save, score_list, adjusted_score_list_all, results_filename, use_xgb=use_xgb)

    # 打印最终结果
    print("\n=== Final Results ===")
    if use_xgb:
        valid_adjusted_scores = [s for s in adjusted_score_list_all if s and len(s) > 0]
        if valid_adjusted_scores:
            print("\nAfter XGBoost Adjustment:")
            if metric != 'all':
                valid_scores = [s[0] for s in valid_adjusted_scores]
                mean_score = np.mean(valid_scores) if valid_scores else 0
                print(f'{metric} mean: {mean_score:.6f}')
            else:
                mlist = ['re', 'mae', 'rmse']
                for i in range(3):
                    valid_scores = [s[i] for s in valid_adjusted_scores if len(s) > i]
                    mean_score = np.mean(valid_scores) if valid_scores else 0
                    print(f'{mlist[i]} mean: {mean_score:.6f}')
    else:
        print("Only Attention Model:")
        if metric != 'all':
            valid_scores = [s[0] for s in score_list if len(s) > 0]
            mean_score = np.mean(valid_scores) if valid_scores else 0
            print(f'{metric} mean: {mean_score:.6f}')
        else:
            mlist = ['re', 'mae', 'rmse']
            for i in range(3):
                valid_scores = [s[i] for s in score_list if len(s) > i]
                mean_score = np.mean(valid_scores) if valid_scores else 0
                print(f'{mlist[i]} mean: {mean_score:.6f}')

    print(f"\nAll results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()