import torch
from src.model import get_unet

def load_ensemble(model_paths, device):
    models = []
    for path in model_paths:
        model = get_unet()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model.to(device)
        models.append(model)
    return models

def ensemble_predict(models, x):
    with torch.no_grad():
        preds = [model(x) for model in models]
    stacked = torch.stack(preds, dim=0)
    avg = torch.mean(stacked, dim=0)
    final = (avg > 0.5).float()
    return final, avg