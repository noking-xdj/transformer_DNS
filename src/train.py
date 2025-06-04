import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import FlowDataset
from model import TransformerModel
from utils import save_checkpoint


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = config['data_dir']
    batch_size = config.get('batch_size', 4)
    seq_len = config.get('sequence_length', 10)
    epochs = config.get('epochs', 10)
    feature_dim = config.get('feature_dim', 128)

    dataset = FlowDataset(data_dir, sequence_length=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(feature_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            # reshape to (seq_len, batch, feature_dim)
            x = x.to(device).permute(1, 0, 2)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: loss={total_loss/len(loader):.6f}')
        save_checkpoint(model, optimizer, epoch, config['save_dir'])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train transformer on DNS data')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(config['save_dir'], exist_ok=True)
    train(config)

if __name__ == '__main__':
    main()
