import torch

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据配置
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir_path = 'datasets/CALCE/'
Rated_Capacity = 1.1

# 模型参数 - 移除MoE相关参数
feature_size = 64
hidden_dim = 32
num_layers = 1
nhead = 4
dropout_att = 0.0
use_xgb = False

# 训练参数
lr = 5e-4
weight_decay = 0.0
EPOCH = 500
window_size = feature_size

# 评估参数
metric = 'all'  # 're', 'mae', 'rmse', or 'all'

# 随机种子
seeds = [0, 1, 2, 3, 4]

# XGBoost参数
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}