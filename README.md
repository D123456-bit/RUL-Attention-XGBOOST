# RUL-Attention-XGBOOST
# 锂电池容量预测系统 - 基于注意力机制的深度学习模型

## 项目简介

本项目实现了一个基于注意力机制（Attention Mechanism）的深度学习模型，用于预测锂电池在循环使用过程中的容量衰减趋势。系统支持纯注意力模型和注意力模型+XGBoost混合调整两种预测方案，能够对锂电池的剩余使用寿命进行准确预测。

## 主要功能

- 🔋 **多电池数据处理**：支持同时处理多个锂电池的测试数据（CS2_35、CS2_36、CS2_37、CS2_38）
- 🧠 **注意力机制模型**：基于自注意力机制（Self-Attention）的时间序列预测模型
- 📊 **XGBoost混合调整**：使用XGBoost对注意力模型预测结果进行误差修正
- 📈 **多指标评估**：提供相对误差（RE）、平均绝对误差（MAE）、均方根误差（RMSE）等多种评估指标
- 🎨 **可视化分析**：自动生成容量衰减曲线、预测对比图等可视化结果

## 项目结构

```
battery_capacity_prediction/
├── config.py              # 配置文件（模型参数、训练参数等）
├── data_loader.py         # 数据加载和预处理
├── model.py               # 注意力机制模型定义
├── trainer_attention.py   # 纯注意力模型训练器
├── trainer_xgboots.py     # 注意力+XGBoost混合训练器
├── xgb_adjuster.py        # XGBoost调整器实现
├── utils.py               # 工具函数（评估指标、随机种子等）
├── main1.py               # 主程序入口
├── datasets/              # 数据集目录
│   └── CALCE/            # CALCE锂电池数据集
│       ├── CS2_35/       # 电池35数据
│       ├── CS2_36/       # 电池36数据
│       ├── CS2_37/       # 电池37数据
│       └── CS2_38/       # 电池38数据
└── results_*/            # 运行结果保存目录（自动生成）
```


### 模型选择

在 `main1.py` 中设置 `use_xgb` 参数：

```python
# 使用纯注意力模型
use_xgb = False

# 使用注意力+XGBoost混合模型
use_xgb = True
```

## 运行方法

### 基本运行

```bash
python main1.py
```

### 运行结果

程序运行后会生成：

1. **容量衰减图**：`capacity_degradation.png` - 所有电池的容量衰减曲线
2. **预测对比图**：每个电池在不同随机种子下的预测值vs实际值对比图
3. **结果文件**：`results_*/results_*.txt` - 包含模型参数和评估指标
4. **控制台输出**：实时显示训练进度和评估结果

### 输出示例

```
=== Processing Battery CS2_35 ===
Train data length: 850
Test data length: 150

--- Training Attention Model on CS2_35 with seed 0 (Run 1/5) ---
Epoch 100/500, Loss: 0.002345
Epoch 200/500, Loss: 0.001234
Attention Model Results - MAE: 0.0234, RMSE: 0.0312, RE: 0.1523

=== Final Results ===
Only Attention Model:
re mean: 0.1523 ± 0.0234
mae mean: 0.0234 ± 0.0031
rmse mean: 0.0312 ± 0.0045

All results and plots saved to: results_20240101_120000
```

