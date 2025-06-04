import torch
from dataset import FlowDataset
from model import TransformerModel
from utils import load_checkpoint


def evaluate(checkpoint_path, data_dir, sequence_length=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FlowDataset(data_dir, sequence_length=sequence_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    checkpoint = load_checkpoint(checkpoint_path)
    model = TransformerModel(checkpoint['feature_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    mse = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).permute(1, 0, 2)
            y = y.to(device)
            pred = model(x)
            loss = mse(pred, y)
            total_loss += loss.item()
    print(f'MSE: {total_loss / len(loader):.6f}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate transformer model')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir)

if __name__ == '__main__':
    main()
