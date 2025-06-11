import torch
import torch.nn as nn
from src.model import get_unet

def dice_loss(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    bce = nn.BCELoss()
    for x, y in loader:
        x, y = x.to(DEVICE, dtype=torch.float), y.to(DEVICE, dtype=torch.float)
        x = x[..., x.shape[-1] // 2]
        y = y[..., y.shape[-1] // 2]  
        if x.shape[1] > 2:
             x = x[:, :2, ...]   
        y = y.unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = 0.5 * bce(out, y) + 0.5 * dice_loss(out, y)
        loss.backward()
        optimizer.step()

def train_ensemble(data_dir, n_models=3, epochs=2, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_dataloaders(data_dir, batch_size)
    for i in range(n_models):
        model = get_unet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            train_one_epoch(model, loader, optimizer, device)
        torch.save(model.state_dict(), f"base_model_{i}.pth")