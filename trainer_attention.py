import numpy as np
import torch
import torch.nn as nn
from data_loader import get_train_test
from utils import setup_seed, evaluation, relative_error


def train_model(Battery, Battery_list, config, model_params, train_params):
    score_list_all, result_list_all = [], []

    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(
            Battery, name, config.window_size)

        score_list_seed, result_list_seed = [], []

        for seed in train_params['seeds']:
            print(f'Training on {name} with seed {seed}')

            setup_seed(seed)
            model = train_params['model_class'](**model_params)
            model = model.to(config.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_params['lr'],
                weight_decay=train_params['weight_decay']
            )
            criterion = nn.MSELoss()

            test_x = train_data.copy()
            y_pred_seed = []
            mae, rmse, re = 1, 1, 1

            # 训练注意力模型
            for epoch in range(train_params['EPOCH']):
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

                if (epoch + 1) % 100 == 0:
                    test_x = train_data.copy()
                    point_list = []

                    while (len(test_x) - len(train_data)) < len(test_data):
                        x = np.reshape(np.array(test_x[-model_params['feature_size']:]) / config.Rated_Capacity,
                                       (-1, 1, model_params['feature_size'])).astype(np.float32)
                        x = torch.from_numpy(x)
                        x = x.to(config.device)
                        pred = model(x)
                        next_point = pred.cpu().data.numpy()[0, 0] * config.Rated_Capacity
                        test_x.append(next_point)
                        point_list.append(next_point)

                    y_pred_seed = point_list
                    mae, rmse = evaluation(y_test=test_data, y_predict=y_pred_seed)
                    re = relative_error(y_test=test_data, y_predict=y_pred_seed,
                                        threshold=config.Rated_Capacity * 0.7)

                    print(
                        f'epoch:{epoch + 1:<3d} | loss:{loss.item():<6.4f} | MAE:{mae:<6.4f} | RMSE:{rmse:<6.4f} | RE:{re:<6.4f}')

                if loss < 1e-3:
                    break

            if train_params['metric'] == 're':
                score = [re]
            elif train_params['metric'] == 'mae':
                score = [mae]
            elif train_params['metric'] == 'rmse':
                score = [rmse]
            else:
                score = [re, mae, rmse]

            score_list_seed.append(score)
            result_list_seed.append(y_pred_seed)

        score_list_all.extend(score_list_seed)
        result_list_all.extend(result_list_seed)

    return score_list_all, result_list_all