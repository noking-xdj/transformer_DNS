import torch  # 张量库 / tensor library
from torch import nn  # 神经网络模块 / neural network modules

class TransformerModel(nn.Module):  # 变换器模型类 / transformer model
    def __init__(self, feature_dim, num_layers=4, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()  # 初始化父类 / init base class
        self.pos_encoder = PositionalEncoding(feature_dim, dropout)  # 位置编码 / positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )  # 单层编码器设置 / one transformer encoder layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 堆叠编码层 / stacked encoder
        self.fc_out = nn.Linear(feature_dim, feature_dim)  # 输出线性层 / output linear layer

    def forward(self, src):  # 前向传播 / forward pass
        # src shape: (seq_len, batch_size, feature_dim) 输入形状说明
        src = self.pos_encoder(src)  # 加入位置编码 / add positional encoding
        output = self.transformer(src)  # 通过编码器 / pass through transformer
        output = self.fc_out(output[-1])  # 取最后一步输出并线性映射 / map last token
        return output  # 返回预测 / return prediction

class PositionalEncoding(nn.Module):  # 位置编码模块 / positional encoding
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()  # 初始化父类
        self.dropout = nn.Dropout(p=dropout)  # 随机失活层 / dropout layer

        position = torch.arange(0, max_len).unsqueeze(1)  # 序列位置 / positions
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # 周期项 / frequency term
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码矩阵 / init encoding
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位使用正弦 / sine on even
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位使用余弦 / cosine on odd
        pe = pe.unsqueeze(1)  # 扩展批次维度 / add batch dimension
        self.register_buffer('pe', pe)  # 缓存为持久缓冲区 / register buffer

    def forward(self, x):  # 前向计算 / forward
        x = x + self.pe[: x.size(0)]  # 加入位置编码 / add encoding
        return self.dropout(x)  # 应用 dropout / apply dropout
