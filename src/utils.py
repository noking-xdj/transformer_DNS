import torch
import os


def save_checkpoint(model, optimizer, epoch, save_dir):
    path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'feature_dim': model.fc_out.in_features,
    }, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu')
