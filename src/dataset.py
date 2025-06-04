import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    """Dataset for DNS flow-field time series."""

    def __init__(self, data_dir, sequence_length=10, transform=None, prepare=False, input_dir=None, output_dir=None):
        if prepare:
            self._prepare_data(input_dir, output_dir)
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

    def _prepare_data(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for fname in os.listdir(input_dir):
            if fname.endswith('.npy'):
                arr = np.load(os.path.join(input_dir, fname))
                np.save(os.path.join(output_dir, fname), arr.astype(np.float32))

    def __len__(self):
        return len(self.files) - self.seq_len

    def __getitem__(self, idx):
        seq = [np.load(self.files[i]) for i in range(idx, idx + self.seq_len + 1)]
        x = np.stack(seq[:-1], axis=0)
        y = seq[-1]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return torch.from_numpy(x), torch.from_numpy(y)

def create_dataloader(data_dir, batch_size=4, sequence_length=10, shuffle=True):
    dataset = FlowDataset(data_dir=data_dir, sequence_length=sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare or inspect DNS dataset')
    parser.add_argument('--prepare', action='store_true', help='prepare raw data')
    parser.add_argument('--input', type=str, help='raw data directory')
    parser.add_argument('--output', type=str, help='processed data directory')
    args = parser.parse_args()
    if args.prepare:
        FlowDataset(data_dir=args.output, prepare=True, input_dir=args.input, output_dir=args.output)
        print(f'Data prepared in {args.output}')
    else:
        dl = create_dataloader(args.input)
        for batch in dl:
            print(batch[0].shape, batch[1].shape)
            break
