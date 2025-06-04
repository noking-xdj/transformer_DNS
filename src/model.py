import torch
from torch import nn

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_layers=4, nhead=8, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(feature_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, src):
        # src shape: (seq_len, batch_size, feature_dim)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        output = self.fc_out(output[-1])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
