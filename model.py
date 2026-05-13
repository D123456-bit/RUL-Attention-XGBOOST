import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_size, hidden_dim, nhead=4, dropout=0.0):
        super(Attention, self).__init__()
        self.query = nn.Linear(feature_size, hidden_dim)
        self.key = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(feature_size, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x):
        query, key, value = self.query(x), self.key(x), self.value(x)
        out, _ = self.attn(query, key, value)
        return out


class AttModel(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=8, num_layers=1, nhead=4, dropout_att=0.,
                 dropout_rate=0.2, device='cpu'):
        super(AttModel, self).__init__()
        self.feature_size, self.hidden_dim = feature_size, hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.cell = Attention(feature_size=feature_size, hidden_dim=hidden_dim, nhead=nhead, dropout=dropout_att)

        # 添加额外的线性层来增强模型能力
        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dropout(x)
        out = self.cell(x)  # (batch_size, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_dim)

        # 通过全连接层
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)  # (batch_size, 1)

        return out