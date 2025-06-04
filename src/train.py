import os  # 操作系统接口 / OS utilities
import yaml  # 读取配置 / read yaml config
import torch  # 深度学习库 / deep learning library
from torch import nn, optim  # 神经网络和优化器 / neural nets and optimizers
from torch.utils.data import DataLoader  # 数据加载器 / dataloader

from dataset import FlowDataset  # 导入数据集 / dataset class
from model import TransformerModel  # 导入模型 / model class
from utils import save_checkpoint  # 保存检查点工具 / checkpoint utility


def train(config):  # 训练过程 / training procedure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备 / choose device
    data_dir = config['data_dir']  # 数据目录 / dataset directory
    batch_size = config.get('batch_size', 4)  # 批大小 / batch size
    seq_len = config.get('sequence_length', 10)  # 序列长度 / sequence length
    epochs = config.get('epochs', 10)  # 训练轮数 / number of epochs
    feature_dim = config.get('feature_dim', 128)  # 特征维度 / feature dimension

    dataset = FlowDataset(data_dir, sequence_length=seq_len)  # 创建数据集 / create dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建加载器 / create dataloader

    model = TransformerModel(feature_dim).to(device)  # 初始化模型并放到设备 / init model
    criterion = nn.MSELoss()  # 均方误差损失 / MSE loss
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))  # Adam 优化器 / Adam optimizer

    for epoch in range(epochs):  # 逐轮训练 / iterate over epochs
        model.train()  # 设置为训练模式 / set training mode
        total_loss = 0.0  # 累计损失 / accumulate loss
        for x, y in loader:  # 遍历批次 / loop batches
            x = x.to(device).permute(1, 0, 2)  # 调整形状为 (seq, batch, feat) / reshape input
            y = y.to(device)  # 目标移到设备 / move target

            optimizer.zero_grad()  # 清空梯度 / reset gradients
            output = model(x)  # 模型前向 / forward pass
            loss = criterion(output, y)  # 计算损失 / compute loss
            loss.backward()  # 反向传播 / backward
            optimizer.step()  # 更新参数 / step optimizer
            total_loss += loss.item()  # 累加损失 / accumulate
        print(f'Epoch {epoch+1}: loss={total_loss/len(loader):.6f}')  # 打印平均损失 / print average loss
        save_checkpoint(model, optimizer, epoch, config['save_dir'])  # 保存模型 / save checkpoint


def main():  # 命令行入口 / command-line entry
    import argparse  # 解析命令行参数 / argument parsing
    parser = argparse.ArgumentParser(description='Train transformer on DNS data')
    parser.add_argument('--config', type=str, required=True)  # 配置文件路径 / config path
    args = parser.parse_args()  # 解析参数
    with open(args.config) as f:  # 读取配置文件 / load config
        config = yaml.safe_load(f)
    os.makedirs(config['save_dir'], exist_ok=True)  # 创建保存目录 / ensure save dir
    train(config)  # 启动训练 / start training

if __name__ == '__main__':
    main()
