import torch  # 深度学习库 / deep learning library
from dataset import FlowDataset  # 数据集类 / dataset class
from model import TransformerModel  # 模型定义 / model definition
from utils import load_checkpoint  # 加载检查点工具 / load checkpoint helper


def evaluate(checkpoint_path, data_dir, sequence_length=10):
    """评估模型性能 / evaluate model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备 / choose device
    dataset = FlowDataset(data_dir, sequence_length=sequence_length)  # 创建数据集 / create dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)  # 加载器 / dataloader

    checkpoint = load_checkpoint(checkpoint_path)  # 读取检查点 / load checkpoint
    model = TransformerModel(checkpoint['feature_dim'])  # 构建模型 / build model
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载权重 / load weights
    model.to(device)  # 移动到设备 / move to device
    model.eval()  # 设置为评估模式 / evaluation mode

    mse = torch.nn.MSELoss()  # MSE 损失函数 / MSE loss
    total_loss = 0.0  # 总损失 / total loss
    with torch.no_grad():  # 禁用梯度 / disable gradients
        for x, y in loader:  # 遍历数据 / iterate samples
            x = x.to(device).permute(1, 0, 2)  # 调整维度 / reshape
            y = y.to(device)  # 目标放到设备 / move target
            pred = model(x)  # 前向预测 / forward pass
            loss = mse(pred, y)  # 计算损失 / compute loss
            total_loss += loss.item()  # 累加损失 / accumulate
    print(f'MSE: {total_loss / len(loader):.6f}')  # 打印平均误差 / print average mse


def main():  # 命令行入口 / command-line entry
    import argparse  # 命令行解析 / argument parser
    parser = argparse.ArgumentParser(description='Evaluate transformer model')
    parser.add_argument('--checkpoint', required=True)  # 模型路径 / checkpoint path
    parser.add_argument('--data-dir', required=True)  # 数据目录 / data dir
    args = parser.parse_args()  # 解析参数 / parse args
    evaluate(args.checkpoint, args.data_dir)  # 调用评估 / run evaluation

if __name__ == '__main__':
    main()
