import os  # 操作系统接口 / OS interface
import numpy as np  # 数值计算库 / numerical operations
import torch  # 主要深度学习库 / main deep learning library
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器 / dataset and dataloader helpers

class FlowDataset(Dataset):  # 自定义的数据集类 / custom dataset class
    """Dataset for DNS flow-field time series.
    DNS 流场时间序列数据集
    """

    def __init__(self, data_dir, sequence_length=10, transform=None, prepare=False, input_dir=None, output_dir=None):
        """初始化数据集 / initialize dataset"""
        if prepare:  # 如果需要预处理 / run preparation if required
            self._prepare_data(input_dir, output_dir)
        self.data_dir = data_dir  # 数据目录 / directory of processed files
        self.seq_len = sequence_length  # 序列长度 / number of frames per sample
        self.transform = transform  # 变换函数 / optional transform
        # 按字典序收集所有 .npy 文件 / gather .npy files in sorted order
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

    def _prepare_data(self, input_dir, output_dir):  # 原始数据预处理 / prepare raw data
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录 / create output dir
        for fname in os.listdir(input_dir):  # 遍历输入目录 / loop over input files
            if fname.endswith('.npy'):  # 仅处理.npy / handle only npy files
                arr = np.load(os.path.join(input_dir, fname))  # 加载文件 / load array
                np.save(os.path.join(output_dir, fname), arr.astype(np.float32))  # 保存为float32 / save processed array

    def __len__(self):  # 样本数量 / number of samples
        return len(self.files) - self.seq_len

    def __getitem__(self, idx):  # 获取单个样本 / fetch one sample
        seq = [np.load(self.files[i]) for i in range(idx, idx + self.seq_len + 1)]  # 加载序列 / load series
        x = np.stack(seq[:-1], axis=0)  # 输入序列 / input frames
        y = seq[-1]  # 目标帧 / target frame
        if self.transform:  # 如果有变换 / apply transform
            x = self.transform(x)
            y = self.transform(y)
        return torch.from_numpy(x), torch.from_numpy(y)  # 返回张量 / return tensors

def create_dataloader(data_dir, batch_size=4, sequence_length=10, shuffle=True):
    """创建用于训练的数据加载器 / build dataloader for training"""
    dataset = FlowDataset(data_dir=data_dir, sequence_length=sequence_length)  # 初始化数据集 / init dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # 构建 DataLoader / create dataloader

if __name__ == '__main__':  # 作为脚本运行时的入口 / entry when run as script
    import argparse  # 解析命令行参数 / parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare or inspect DNS dataset')
    parser.add_argument('--prepare', action='store_true', help='prepare raw data')  # 准备数据标志 / flag for preparation
    parser.add_argument('--input', type=str, help='raw data directory')  # 输入目录 / input directory
    parser.add_argument('--output', type=str, help='processed data directory')  # 输出目录 / output directory
    args = parser.parse_args()  # 解析参数 / parse args
    if args.prepare:  # 如果指定 --prepare / if preparation requested
        FlowDataset(data_dir=args.output, prepare=True, input_dir=args.input, output_dir=args.output)  # 调用预处理 / call prepare
        print(f'Data prepared in {args.output}')  # 提示完成 / notify user
    else:  # 否则仅测试加载 / otherwise test loading
        dl = create_dataloader(args.input)
        for batch in dl:
            print(batch[0].shape, batch[1].shape)  # 输出形状 / show shapes
            break
