import torch  # 深度学习库 / deep learning library
import os  # 文件路径操作 / file system operations


def save_checkpoint(model, optimizer, epoch, save_dir):
    """保存模型检查点 / save model checkpoint"""
    path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')  # 构建保存路径 / build path
    torch.save(
        {
            'model_state_dict': model.state_dict(),  # 模型参数 / model weights
            'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态 / optimizer state
            'epoch': epoch,  # 当前轮数 / current epoch
            'feature_dim': model.fc_out.in_features,  # 模型特征维度 / feature dimension
        },
        path,
    )  # 执行保存 / perform save


def load_checkpoint(path):
    """加载模型检查点 / load model checkpoint"""
    return torch.load(path, map_location='cpu')  # 从文件中读取 / load from file
