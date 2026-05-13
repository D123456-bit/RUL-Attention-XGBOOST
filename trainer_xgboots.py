# import numpy as np
# import torch
# import torch.nn as nn
# from data_loader import get_train_test
# from utils import setup_seed, evaluation, relative_error
# from xgb_adjuster import XGBAdjuster
#
#
# def train_attention_model(Battery, Battery_list, config, model_params, train_params):
#     """
#     仅使用注意力模型进行训练和预测
#     """
#     score_list_all, result_list_all = [], []
#
#     for i in range(4):
#         name = Battery_list[i]
#         train_x, train_y, train_data, test_data = get_train_test(
#             Battery, name, config.window_size)
#
#         print(f"\n=== Processing Battery {name} ===")
#         print(f"Train data length: {len(train_data)}")
#         print(f"Test data length: {len(test_data)}")
#
#         score_list_seed, result_list_seed = [], []
#
#         for seed_idx, seed in enumerate(train_params['seeds']):
#             print(
#                 f'\n--- Training Attention Model on {name} with seed {seed} (Run {seed_idx + 1}/{len(train_params["seeds"])}) ---')
#
#             setup_seed(seed)
#             model = train_params['model_class'](**model_params)
#             model = model.to(config.device)
#
#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=train_params['lr'],
#                 weight_decay=train_params['weight_decay']
#             )
#             criterion = nn.MSELoss()
#
#             # 添加学习率调度和早停
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', patience=50, factor=0.5, verbose=False
#             )
#
#             best_loss = float('inf')
#             patience = 100
#             patience_counter = 0
#             best_model_state = None
#
#             # 训练注意力模型
#             for epoch in range(train_params['EPOCH']):
#                 model.train()
#
#                 X = np.reshape(train_x / config.Rated_Capacity, (-1, 1, model_params['feature_size'])).astype(
#                     np.float32)
#                 y = np.reshape(train_y[:, -1] / config.Rated_Capacity, (-1, 1)).astype(np.float32)
#
#                 X, y = torch.from_numpy(X), torch.from_numpy(y)
#                 X, y = X.to(config.device), y.to(config.device)
#
#                 output = model(X)
#                 output = output.reshape(-1, 1)
#                 loss = criterion(output, y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#                 scheduler.step(loss)
#
#                 # 早停机制
#                 if loss.item() < best_loss:
#                     best_loss = loss.item()
#                     patience_counter = 0
#                     best_model_state = model.state_dict().copy()
#                 else:
#                     patience_counter += 1
#
#                 # 每100个epoch显示进度
#                 if (epoch + 1) % 100 == 0:
#                     print(f'Epoch {epoch + 1}/{train_params["EPOCH"]}, Loss: {loss.item():.6f}')
#
#                 if patience_counter >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     model.load_state_dict(best_model_state)
#                     break
#
#                 if loss < 1e-4:  # 更严格的停止条件
#                     print(f"Converged at epoch {epoch + 1}")
#                     break
#
#             # 使用最佳模型生成最终预测
#             if best_model_state:
#                 model.load_state_dict(best_model_state)
#
#             model.eval()
#             attention_predictions = generate_attention_predictions(
#                 model, train_data, test_data, config, model_params
#             )
#
#             # 计算指标
#             if len(attention_predictions) == len(test_data):
#                 mae, rmse = evaluation(y_test=test_data, y_predict=attention_predictions)
#                 re = relative_error(y_test=test_data, y_predict=attention_predictions,
#                                     threshold=config.Rated_Capacity * 0.7)
#
#                 print(f"Attention Model Results - MAE: {mae:<6.4f}, RMSE: {rmse:<6.4f}, RE: {re:<6.4f}")
#
#                 if train_params['metric'] == 're':
#                     score = [re]
#                 elif train_params['metric'] == 'mae':
#                     score = [mae]
#                 elif train_params['metric'] == 'rmse':
#                     score = [rmse]
#                 else:
#                     score = [re, mae, rmse]
#             else:
#                 print(f"Error: Predictions length mismatch")
#                 score = [1.0, config.Rated_Capacity * 0.1, config.Rated_Capacity * 0.15]
#
#             score_list_seed.append(score)
#             result_list_seed.append(attention_predictions)
#
#         score_list_all.extend(score_list_seed)
#         result_list_all.extend(result_list_seed)
#
#     return score_list_all, result_list_all, []  # 返回空的adjusted_score_list_all
#
#
# def train_attention_with_xgb_adjustment(Battery, Battery_list, config, model_params, train_params, xgb_params):
#     """
#     使用注意力模型进行预测，然后用XGBoost进行调整
#     """
#     score_list_all, result_list_all = [], []
#     adjusted_score_list_all = []
#
#     for i in range(4):
#         name = Battery_list[i]
#         train_x, train_y, train_data, test_data = get_train_test(
#             Battery, name, config.window_size)
#
#         print(f"\n=== Processing Battery {name} ===")
#         print(f"Train data length: {len(train_data)}")
#         print(f"Test data length: {len(test_data)}")
#
#         score_list_seed, result_list_seed = [], []
#         adjusted_score_list_seed = []
#
#         for seed_idx, seed in enumerate(train_params['seeds']):
#             print(
#                 f'\n--- Training Attention+XGB on {name} with seed {seed} (Run {seed_idx + 1}/{len(train_params["seeds"])}) ---')
#
#             setup_seed(seed)
#             model = train_params['model_class'](**model_params)
#             model = model.to(config.device)
#
#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=train_params['lr'],
#                 weight_decay=train_params['weight_decay']
#             )
#             criterion = nn.MSELoss()
#
#             # 训练注意力模型（简化版，专注于训练）
#             best_loss = float('inf')
#             for epoch in range(train_params['EPOCH']):
#                 model.train()
#
#                 X = np.reshape(train_x / config.Rated_Capacity, (-1, 1, model_params['feature_size'])).astype(
#                     np.float32)
#                 y = np.reshape(train_y[:, -1] / config.Rated_Capacity, (-1, 1)).astype(np.float32)
#
#                 X, y = torch.from_numpy(X), torch.from_numpy(y)
#                 X, y = X.to(config.device), y.to(config.device)
#
#                 output = model(X)
#                 output = output.reshape(-1, 1)
#                 loss = criterion(output, y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 if loss.item() < best_loss:
#                     best_loss = loss.item()
#
#                 if (epoch + 1) % 100 == 0:
#                     print(f'Epoch {epoch + 1}/{train_params["EPOCH"]}, Loss: {loss.item():.6f}')
#
#                 if loss < 1e-4:
#                     break
#
#             # 生成注意力模型预测
#             model.eval()
#             attention_predictions = generate_attention_predictions(
#                 model, train_data, test_data, config, model_params
#             )
#
#             # 计算原始注意力模型指标
#             if len(attention_predictions) == len(test_data):
#                 mae, rmse = evaluation(y_test=test_data, y_predict=attention_predictions)
#                 re = relative_error(y_test=test_data, y_predict=attention_predictions,
#                                     threshold=config.Rated_Capacity * 0.7)
#
#                 print(f"Original Attention Model - MAE: {mae:<6.4f}, RMSE: {rmse:<6.4f}, RE: {re:<6.4f}")
#
#                 if train_params['metric'] == 're':
#                     original_score = [re]
#                 elif train_params['metric'] == 'mae':
#                     original_score = [mae]
#                 elif train_params['metric'] == 'rmse':
#                     original_score = [rmse]
#                 else:
#                     original_score = [re, mae, rmse]
#             else:
#                 print(f"Error: Attention predictions length mismatch")
#                 original_score = [1.0, config.Rated_Capacity * 0.1, config.Rated_Capacity * 0.15]
#
#             score_list_seed.append(original_score)
#             result_list_seed.append(attention_predictions)
#
#             # XGBoost调整部分
#             adjusted_predictions = attention_predictions.copy()
#             adjusted_score = []
#
#             if len(attention_predictions) == len(test_data):
#                 print("\nStarting XGBoost adjustment...")
#                 xgb_adjuster = XGBAdjuster(xgb_params)
#
#                 # 使用其他电池数据训练初始XGBoost模型
#                 train_success = xgb_adjuster.train_on_other_batteries(
#                     Battery, Battery_list, name, model, config, model_params
#                 )
#
#                 if train_success:
#                     try:
#                         adjusted_result = xgb_adjuster.incremental_adjust(
#                             attention_predictions=attention_predictions,
#                             historical_data=train_data.copy(),
#                             actual_data_so_far=None
#                         )
#
#                         if adjusted_result is not None and len(adjusted_result) == len(test_data):
#                             adjusted_predictions = adjusted_result
#                             print("XGBoost adjustment completed successfully")
#
#                             # 计算调整后的指标
#                             mae_adj, rmse_adj = evaluation(y_test=test_data, y_predict=adjusted_predictions)
#                             re_adj = relative_error(y_test=test_data, y_predict=adjusted_predictions,
#                                                     threshold=config.Rated_Capacity * 0.7)
#
#                             print(
#                                 f"After XGBoost Adjustment - MAE: {mae_adj:<6.4f}, RMSE: {rmse_adj:<6.4f}, RE: {re_adj:<6.4f}")
#
#                             # 评估调整效果
#                             xgb_adjuster.evaluate_adjustment(
#                                 original_predictions=attention_predictions,
#                                 adjusted_predictions=adjusted_predictions,
#                                 actual_values=test_data
#                             )
#
#                             if train_params['metric'] == 're':
#                                 adjusted_score = [re_adj]
#                             elif train_params['metric'] == 'mae':
#                                 adjusted_score = [mae_adj]
#                             elif train_params['metric'] == 'rmse':
#                                 adjusted_score = [rmse_adj]
#                             else:
#                                 adjusted_score = [re_adj, mae_adj, rmse_adj]
#                         else:
#                             print("XGBoost adjustment failed, using original predictions")
#                             adjusted_score = original_score
#
#                     except Exception as e:
#                         print(f"XGBoost adjustment error: {e}")
#                         print("Using original predictions")
#                         adjusted_score = original_score
#                 else:
#                     print("XGBoost training failed, using original predictions")
#                     adjusted_score = original_score
#             else:
#                 print("Skipping XGBoost adjustment due to invalid predictions")
#                 adjusted_score = original_score
#
#             adjusted_score_list_seed.append(adjusted_score)
#
#         score_list_all.extend(score_list_seed)
#         result_list_all.extend(result_list_seed)
#         adjusted_score_list_all.extend(adjusted_score_list_seed)
#
#     return score_list_all, result_list_all, adjusted_score_list_all
#
#
# def generate_attention_predictions(model, train_data, test_data, config, model_params):
#     """
#     生成注意力模型的预测结果
#     """
#     predictions = []
#     historical_data = train_data.copy()
#
#     print(f"Generating {len(test_data)} predictions with attention model...")
#
#     for i in range(len(test_data)):
#         # 确保有足够的历史数据
#         if len(historical_data) < model_params['feature_size']:
#             needed = model_params['feature_size'] - len(historical_data)
#             padding = train_data[-needed:] if len(train_data) >= needed else train_data
#             historical_data = padding + historical_data
#
#         # 准备输入数据
#         x_input = np.reshape(
#             np.array(historical_data[-model_params['feature_size']:]) / config.Rated_Capacity,
#             (1, 1, model_params['feature_size'])
#         ).astype(np.float32)
#
#         x_input = torch.from_numpy(x_input).to(config.device)
#
#         # 预测
#         with torch.no_grad():
#             pred = model(x_input)
#             next_point = pred.cpu().numpy()[0, 0] * config.Rated_Capacity
#
#         # 处理异常值
#         if np.isnan(next_point) or np.isinf(next_point):
#             next_point = historical_data[-1] if historical_data else config.Rated_Capacity * 0.8
#
#         predictions.append(next_point)
#         historical_data.append(next_point)
#
#         if (i + 1) % 100 == 0 and (i + 1) < len(test_data):
#             print(f"Generated {i + 1}/{len(test_data)} predictions")
#
#     print(f"Prediction generation completed: {len(predictions)} points")
#     return predictions
#
#
# def train_model(Battery, Battery_list, config, model_params, train_params, xgb_params=None):
#     """
#     统一的训练函数，根据参数选择使用哪种模型
#     """
#     model_type = train_params.get('model_type', 'attention')  # 默认使用attention
#
#     if model_type == 'xgb':
#         print("=" * 60)
#         print("Training with Attention + XGBoost Adjustment")
#         print("=" * 60)
#         return train_attention_with_xgb_adjustment(
#             Battery, Battery_list, config, model_params, train_params, xgb_params
#         )
#     else:
#         print("=" * 60)
#         print("Training with Attention Model Only")
#         print("=" * 60)
#         return train_attention_model(
#             Battery, Battery_list, config, model_params, train_params
#         )
import numpy as np
import torch
import torch.nn as nn
from data_loader import get_train_test
from utils import setup_seed, evaluation, relative_error
from xgb_adjuster import XGBAdjuster
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
def plot_predictions_vs_actual(test_data, predictions, battery_name, save_path=None):
    """
    绘制预测值与实际值的对比图
    """
    plt.figure(figsize=(12, 6))

    # 生成x轴（循环次数）
    cycles = range(len(test_data))

    # 绘制实际值
    plt.plot(cycles, test_data, 'b-', linewidth=2, label='Actual Capacity', marker='o', markersize=3)

    # 绘制预测值
    plt.plot(cycles, predictions, 'r--', linewidth=2, label='Predicted Capacity', marker='s', markersize=3)

    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title(f'Capacity Prediction vs Actual - Battery {battery_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加误差统计
    mae = mean_absolute_error(test_data, predictions)
    rmse = sqrt(mean_squared_error(test_data, predictions))

    plt.text(0.02, 0.98, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()



def train_attention_model(Battery, Battery_list, config, model_params, train_params):
    """
    仅使用注意力模型进行训练和预测
    """
    score_list_all, result_list_all, test_data_list_all = [], [], []  # 新增：保存测试数据

    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            Battery, name, config.window_size)

        print(f"\n=== Processing Battery {name} ===")
        print(f"Train data length: {len(train_data)}")
        print(f"Test data length: {len(test_data)}")

        score_list_seed, result_list_seed,test_data_list_seed = [], [], []

        for seed_idx, seed in enumerate(train_params['seeds']):
            print(
                f'\n--- Training Attention Model on {name} with seed {seed} (Run {seed_idx + 1}/{len(train_params["seeds"])}) ---')

            setup_seed(seed)
            model = train_params['model_class'](**model_params)
            model = model.to(config.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_params['lr'],
                weight_decay=train_params['weight_decay']
            )
            criterion = nn.MSELoss()

            # 添加学习率调度和早停
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=50, factor=0.5, verbose=False
            )

            best_loss = float('inf')
            patience = 100
            patience_counter = 0
            best_model_state = None

            # 训练注意力模型
            for epoch in range(train_params['EPOCH']):
                model.train()

                X = np.reshape(train_x / config.Rated_Capacity, (-1, 1, model_params['feature_size'])).astype(
                    np.float32)
                y = np.reshape(train_y[:, -1] / config.Rated_Capacity, (-1, 1)).astype(np.float32)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.to(config.device), y.to(config.device)

                output = model(X)
                output = output.reshape(-1, 1)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                scheduler.step(loss)

                # 早停机制
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                # 每100个epoch显示进度
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{train_params["EPOCH"]}, Loss: {loss.item():.6f}')

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_model_state)
                    break

                if loss < 1e-4:  # 更严格的停止条件
                    print(f"Converged at epoch {epoch + 1}")
                    break

            # 使用最佳模型生成最终预测
            if best_model_state:
                model.load_state_dict(best_model_state)

            model.eval()
            attention_predictions = generate_attention_predictions(
                model, train_data, test_data, config, model_params
            )

            # 计算指标
            if len(attention_predictions) == len(test_data):
                mae, rmse = evaluation(y_test=test_data, y_predict=attention_predictions)
                re = relative_error(y_test=test_data, y_predict=attention_predictions,
                                    threshold=config.Rated_Capacity * 0.7)

                print(f"Attention Model Results - MAE: {mae:<6.4f}, RMSE: {rmse:<6.4f}, RE: {re:<6.4f}")

                if train_params['metric'] == 're':
                    score = [re]
                elif train_params['metric'] == 'mae':
                    score = [mae]
                elif train_params['metric'] == 'rmse':
                    score = [rmse]
                else:
                    score = [re, mae, rmse]
            else:
                print(f"Error: Predictions length mismatch")
                score = [1.0, config.Rated_Capacity * 0.1, config.Rated_Capacity * 0.15]

            score_list_seed.append(score)
            result_list_seed.append(attention_predictions)
            test_data_list_seed.append(test_data)  # 保存测试数据

        score_list_all.extend(score_list_seed)
        result_list_all.extend(result_list_seed)
        test_data_list_all.extend(test_data_list_seed)

    return score_list_all, result_list_all, test_data_list_all  # 返回测试数据


def train_attention_with_xgb_adjustment(Battery, Battery_list, config, model_params, train_params, xgb_params):
    """
    使用注意力模型进行预测，然后用XGBoost进行调整
    """
    score_list_all, result_list_all, adjusted_score_list_all, test_data_list_all = [], [], [], []

    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            Battery, name, config.window_size)

        print(f"\n=== Processing Battery {name} ===")
        print(f"Train data length: {len(train_data)}")
        print(f"Test data length: {len(test_data)}")

        score_list_seed, result_list_seed = [], []
        adjusted_score_list_seed,test_data_list_seed = [], []

        for seed_idx, seed in enumerate(train_params['seeds']):
            print(
                f'\n--- Training Attention+XGB on {name} with seed {seed} (Run {seed_idx + 1}/{len(train_params["seeds"])}) ---')

            setup_seed(seed)
            model = train_params['model_class'](**model_params)
            model = model.to(config.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_params['lr'],
                weight_decay=train_params['weight_decay']
            )
            criterion = nn.MSELoss()

            # 训练注意力模型（简化版，专注于训练）
            best_loss = float('inf')
            for epoch in range(train_params['EPOCH']):
                model.train()

                X = np.reshape(train_x / config.Rated_Capacity, (-1, 1, model_params['feature_size'])).astype(
                    np.float32)
                y = np.reshape(train_y[:, -1] / config.Rated_Capacity, (-1, 1)).astype(np.float32)

                X, y = torch.from_numpy(X), torch.from_numpy(y)
                X, y = X.to(config.device), y.to(config.device)

                output = model(X)
                output = output.reshape(-1, 1)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()

                if (epoch + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{train_params["EPOCH"]}, Loss: {loss.item():.6f}')

                if loss < 1e-4:
                    break

            # 生成注意力模型预测
            model.eval()
            attention_predictions = generate_attention_predictions(
                model, train_data, test_data, config, model_params
            )

            # 计算原始注意力模型指标
            if len(attention_predictions) == len(test_data):
                mae, rmse = evaluation(y_test=test_data, y_predict=attention_predictions)
                re = relative_error(y_test=test_data, y_predict=attention_predictions,
                                    threshold=config.Rated_Capacity * 0.7)

                print(f"Original Attention Model - MAE: {mae:<6.4f}, RMSE: {rmse:<6.4f}, RE: {re:<6.4f}")

                if train_params['metric'] == 're':
                    original_score = [re]
                elif train_params['metric'] == 'mae':
                    original_score = [mae]
                elif train_params['metric'] == 'rmse':
                    original_score = [rmse]
                else:
                    original_score = [re, mae, rmse]
            else:
                print(f"Error: Attention predictions length mismatch")
                original_score = [1.0, config.Rated_Capacity * 0.1, config.Rated_Capacity * 0.15]

            score_list_seed.append(original_score)
            result_list_seed.append(attention_predictions)

            # XGBoost调整部分
            adjusted_predictions = attention_predictions.copy()
            adjusted_score = []

            if len(attention_predictions) == len(test_data):
                print("\nStarting XGBoost adjustment...")
                xgb_adjuster = XGBAdjuster(xgb_params)

                # 使用其他电池数据训练初始XGBoost模型
                train_success = xgb_adjuster.train_on_other_batteries(
                    Battery, Battery_list, name, model, config, model_params
                )

                if train_success:
                    try:
                        adjusted_result = xgb_adjuster.incremental_adjust(
                            attention_predictions=attention_predictions,
                            historical_data=train_data.copy(),
                            actual_data_so_far=None
                        )

                        if adjusted_result is not None and len(adjusted_result) == len(test_data):
                            adjusted_predictions = adjusted_result
                            print("XGBoost adjustment completed successfully")

                            # 计算调整后的指标
                            mae_adj, rmse_adj = evaluation(y_test=test_data, y_predict=adjusted_predictions)
                            re_adj = relative_error(y_test=test_data, y_predict=adjusted_predictions,
                                                    threshold=config.Rated_Capacity * 0.7)

                            print(
                                f"After XGBoost Adjustment - MAE: {mae_adj:<6.4f}, RMSE: {rmse_adj:<6.4f}, RE: {re_adj:<6.4f}")

                            # 评估调整效果
                            xgb_adjuster.evaluate_adjustment(
                                original_predictions=attention_predictions,
                                adjusted_predictions=adjusted_predictions,
                                actual_values=test_data
                            )

                            if train_params['metric'] == 're':
                                adjusted_score = [re_adj]
                            elif train_params['metric'] == 'mae':
                                adjusted_score = [mae_adj]
                            elif train_params['metric'] == 'rmse':
                                adjusted_score = [rmse_adj]
                            else:
                                adjusted_score = [re_adj, mae_adj, rmse_adj]
                        else:
                            print("XGBoost adjustment failed, using original predictions")
                            adjusted_score = original_score

                    except Exception as e:
                        print(f"XGBoost adjustment error: {e}")
                        print("Using original predictions")
                        adjusted_score = original_score
                else:
                    print("XGBoost training failed, using original predictions")
                    adjusted_score = original_score
            else:
                print("Skipping XGBoost adjustment due to invalid predictions")
                adjusted_score = original_score
            test_data_list_seed.append(test_data)  # 保存测试数据
            adjusted_score_list_seed.append(adjusted_score)

        score_list_all.extend(score_list_seed)
        result_list_all.extend(result_list_seed)
        adjusted_score_list_all.extend(adjusted_score_list_seed)
        test_data_list_all.extend(test_data_list_seed)

    return score_list_all, result_list_all, adjusted_score_list_all, test_data_list_all



def generate_attention_predictions(model, train_data, test_data, config, model_params):
    """
    生成注意力模型的预测结果
    """
    predictions = []
    historical_data = train_data.copy()

    print(f"Generating {len(test_data)} predictions with attention model...")

    for i in range(len(test_data)):
        # 确保有足够的历史数据
        if len(historical_data) < model_params['feature_size']:
            needed = model_params['feature_size'] - len(historical_data)
            padding = train_data[-needed:] if len(train_data) >= needed else train_data
            historical_data = padding + historical_data

        # 准备输入数据
        x_input = np.reshape(
            np.array(historical_data[-model_params['feature_size']:]) / config.Rated_Capacity,
            (1, 1, model_params['feature_size'])
        ).astype(np.float32)

        x_input = torch.from_numpy(x_input).to(config.device)

        # 预测
        with torch.no_grad():
            pred = model(x_input)
            next_point = pred.cpu().numpy()[0, 0] * config.Rated_Capacity

        # 处理异常值
        if np.isnan(next_point) or np.isinf(next_point):
            next_point = historical_data[-1] if historical_data else config.Rated_Capacity * 0.8

        predictions.append(next_point)
        historical_data.append(next_point)

        if (i + 1) % 100 == 0 and (i + 1) < len(test_data):
            print(f"Generated {i + 1}/{len(test_data)} predictions")

    print(f"Prediction generation completed: {len(predictions)} points")
    return predictions


def train_model(Battery, Battery_list, config, model_params, train_params, xgb_params=None):
    """
    统一的训练函数，根据参数选择使用哪种模型
    """
    model_type = train_params.get('model_type', 'attention')  # 默认使用attention

    if model_type == 'xgb':
        print("=" * 60)
        print("Training with Attention + XGBoost Adjustment")
        print("=" * 60)
        score_list, result_list, adjusted_score_list, test_data_list = train_attention_with_xgb_adjustment(
            Battery, Battery_list, config, model_params, train_params, xgb_params
        )
        return score_list, result_list, adjusted_score_list, test_data_list
    else:
        print("=" * 60)
        print("Training with Attention Model Only")
        print("=" * 60)
        score_list, result_list, test_data_list = train_attention_model(
            Battery, Battery_list, config, model_params, train_params
        )
        return score_list, result_list, [], test_data_list  # 返回空列表保持一致性