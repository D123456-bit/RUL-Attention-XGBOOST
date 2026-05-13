import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import torch

class XGBAdjuster:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'max_depth': 4,  # 减小深度以适应小样本
                'learning_rate': 0.1,
                'n_estimators': 50,  # 减少树的数量
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        else:
            self.params = params

        self.model = None
        self.is_trained = False
        self.feature_length = 13  # 固定特征长度

    def prepare_single_feature(self, attention_pred, historical_data):
        """
        为单个预测点准备特征
        """
        feature_vector = []

        # 1. 注意力模型预测值
        feature_vector.append(float(attention_pred))

        # 2. 历史统计特征 (7个)
        if len(historical_data) > 0:
            hist_data = np.array(historical_data, dtype=np.float64)
            feature_vector.extend([
                float(np.mean(hist_data)),  # 历史均值
                float(np.std(hist_data)) if len(hist_data) > 1 else 0.0,  # 历史标准差
                float(np.min(hist_data)),  # 历史最小值
                float(np.max(hist_data)),  # 历史最大值
                float(hist_data[-1] - hist_data[0]) if len(hist_data) > 1 else 0.0,  # 历史变化量
                float((hist_data[-1] - hist_data[0]) / len(hist_data)) if len(hist_data) > 1 else 0.0,  # 平均变化率
                float(len(hist_data))  # 历史数据长度
            ])
        else:
            # 如果没有历史数据，使用默认值
            feature_vector.extend([float(attention_pred)] * 7)

        # 3. 最近的历史数据点 (5个)
        if len(historical_data) >= 5:
            # 取最近的5个点
            recent_points = historical_data[-5:]
            feature_vector.extend([float(x) for x in recent_points])
        elif len(historical_data) > 0:
            # 如果历史点不足5个，用最后一个点填充
            recent_points = list(historical_data)
            while len(recent_points) < 5:
                recent_points.append(historical_data[-1])
            feature_vector.extend([float(x) for x in recent_points])
        else:
            # 如果没有历史数据，用预测值填充
            feature_vector.extend([float(attention_pred)] * 5)

        # 确保特征向量长度正确
        if len(feature_vector) != self.feature_length:
            if len(feature_vector) > self.feature_length:
                feature_vector = feature_vector[:self.feature_length]
            else:
                feature_vector.extend([float(attention_pred)] * (self.feature_length - len(feature_vector)))

        return np.array(feature_vector, dtype=np.float64).reshape(1, -1)

    def train_on_other_batteries(self, Battery, Battery_list, current_battery, model, config, model_params):
        """
        使用其他电池数据训练初始XGBoost模型
        """
        print("Training initial XGBoost model on other batteries...")

        all_features = []
        all_targets = []

        for name in Battery_list:
            if name != current_battery:
                print(f"Processing battery {name} for training data...")
                train_x, train_y, train_data, test_data = Battery[name]['capacity'], None, Battery[name][
                    'capacity'].tolist(), []

                # 使用注意力模型在其他电池数据上生成预测
                if len(train_data) > model_params['feature_size'] + 10:
                    # 生成训练样本：使用滑动窗口
                    for i in range(model_params['feature_size'], len(train_data) - 1):
                        # 准备输入数据
                        input_seq = train_data[i - model_params['feature_size']:i]
                        actual_value = train_data[i]

                        # 使用注意力模型预测
                        X = np.reshape(np.array(input_seq) / config.Rated_Capacity,
                                       (1, 1, model_params['feature_size'])).astype(np.float32)
                        X = torch.from_numpy(X).to(config.device)

                        with torch.no_grad():
                            pred = model(X)
                            attention_pred = pred.cpu().numpy()[0, 0] * config.Rated_Capacity

                        # 准备特征
                        historical_data = train_data[:i]  # 到当前点的所有历史数据
                        features = self.prepare_single_feature(attention_pred, historical_data)

                        all_features.append(features[0])
                        all_targets.append(actual_value)

        if len(all_features) > 10:
            X_train = np.array(all_features, dtype=np.float64)
            y_train = np.array(all_targets, dtype=np.float64)

            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X_train, y_train)
            self.is_trained = True

            print(f"Initial XGBoost model trained with {len(X_train)} samples")
            return True
        else:
            print("Not enough training data from other batteries")
            return False

    def incremental_adjust(self, attention_predictions, historical_data, actual_data_so_far=None):
        """
        逐步调整预测：对每个测试点重新训练XGBoost
        """

        print(f"Debug: incremental_adjust called with {len(attention_predictions)} attention predictions")
        print(f"Debug: historical_data length: {len(historical_data)}")

        if not self.is_trained or self.model is None:
            print("XGBoost model not trained, returning original predictions")
            return attention_predictions

        adjusted_predictions = []
        current_historical = historical_data.copy()

        print("Starting incremental adjustment...")

        for i, att_pred in enumerate(attention_predictions):
            if i % 50 == 0:  # 每50个点打印一次进度
                print(f"Adjusting point {i + 1}/{len(attention_predictions)}")

            # 准备当前点的特征
            features = self.prepare_single_feature(att_pred, current_historical)

            # 使用当前模型调整预测
            adjusted_pred = self.model.predict(features)[0]
            adjusted_predictions.append(adjusted_pred)

            # 如果有实际数据，更新模型
            if actual_data_so_far is not None and i < len(actual_data_so_far):
                actual_value = actual_data_so_far[i]

                # 在线学习：用新数据更新模型
                try:
                    # 准备训练数据：当前特征和实际值
                    X_update = features
                    y_update = np.array([actual_value], dtype=np.float64)

                    # 在线更新模型（使用partial_fit如果支持，否则重新训练）
                    if hasattr(self.model, 'partial_fit'):
                        self.model.partial_fit(X_update, y_update)
                    else:
                        # 小批量重新训练：积累一些数据后重新训练
                        if hasattr(self, 'update_buffer'):
                            self.update_buffer['X'].append(features[0])
                            self.update_buffer['y'].append(actual_value)
                        else:
                            self.update_buffer = {'X': [features[0]], 'y': [actual_value]}

                        # 每积累10个点重新训练一次
                        if len(self.update_buffer['X']) >= 10:
                            X_retrain = np.array(self.update_buffer['X'], dtype=np.float64)
                            y_retrain = np.array(self.update_buffer['y'], dtype=np.float64)
                            self.model.fit(X_retrain, y_retrain)
                            self.update_buffer = {'X': [], 'y': []}
                            print(f"Model updated with new data at point {i + 1}")

                except Exception as e:
                    print(f"Error updating model at point {i + 1}: {e}")

            # 将调整后的预测值加入到历史数据中（用于下一个点的预测）
            current_historical.append(adjusted_pred)

        # 在返回前检查结果
        if adjusted_predictions is not None:
            print(f"Debug: returning {len(adjusted_predictions)} adjusted predictions")
        else:
            print("Debug: returning None for adjusted predictions")

        # return adjusted_predictions
        print("Incremental adjustment completed")
        return adjusted_predictions

    def evaluate_adjustment(self, original_predictions, adjusted_predictions, actual_values):
        """评估调整效果"""
        try:
            original_predictions = np.array(original_predictions, dtype=np.float64)
            adjusted_predictions = np.array(adjusted_predictions, dtype=np.float64)
            actual_values = np.array(actual_values, dtype=np.float64)

            min_len = min(len(original_predictions), len(adjusted_predictions), len(actual_values))

            if min_len == 0:
                print("Warning: No data available for evaluation")
                return {
                    'original_mae': 0, 'adjusted_mae': 0,
                    'original_rmse': 0, 'adjusted_rmse': 0,
                    'improvement_mae': 0, 'improvement_rmse': 0
                }

            orig_pred = original_predictions[:min_len]
            adj_pred = adjusted_predictions[:min_len]
            actual = actual_values[:min_len]

            original_mae = mean_absolute_error(actual, orig_pred)
            adjusted_mae = mean_absolute_error(actual, adj_pred)

            original_rmse = sqrt(mean_squared_error(actual, orig_pred))
            adjusted_rmse = sqrt(mean_squared_error(actual, adj_pred))

            improvement_mae = (original_mae - adjusted_mae) / original_mae * 100 if original_mae > 0 else 0
            improvement_rmse = (original_rmse - adjusted_rmse) / original_rmse * 100 if original_rmse > 0 else 0

            print(f"Adjustment Evaluation (using {min_len} samples):")
            print(f"Original - MAE: {original_mae:.4f}, RMSE: {original_rmse:.4f}")
            print(f"Adjusted - MAE: {adjusted_mae:.4f}, RMSE: {adjusted_rmse:.4f}")
            print(f"Improvement - MAE: {improvement_mae:.2f}%, RMSE: {improvement_rmse:.2f}%")

            return {
                'original_mae': original_mae, 'adjusted_mae': adjusted_mae,
                'original_rmse': original_rmse, 'adjusted_rmse': adjusted_rmse,
                'improvement_mae': improvement_mae, 'improvement_rmse': improvement_rmse
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                'original_mae': 0, 'adjusted_mae': 0,
                'original_rmse': 0, 'adjusted_rmse': 0,
                'improvement_mae': 0, 'improvement_rmse': 0
            }