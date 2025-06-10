# File: /isles22-ensemble-segmentation/isles22-ensemble-segmentation/src/predict.py

import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dataset import ISLESDataset
from ensemble import EnsembleSegmentationModel
from utils import load_models

def predict_and_show_ensemble(data_folder, model_paths, idx=0, slice_idx=None):
    dataset = ISLESDataset(data_folder)
    if len(dataset) == 0:
        print("No samples found.")
        return
    
    models = load_models(model_paths)
    ensemble_model = EnsembleSegmentationModel(models)
    
    x, y = dataset[idx]
    x = x.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        pred = ensemble_model(x)
    
    pred_np = pred.squeeze().numpy()
    flair_np = x[0, 1].numpy()
    mask_np = y.numpy()
    
    # Choose a slice index (center if not specified)
    if slice_idx is None:
        slice_idx = flair_np.shape[2] // 2
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("FLAIR (center slice)")
    plt.imshow(flair_np[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_np[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Ensemble Predicted Mask")
    plt.imshow(pred_np[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    predict_and_show_ensemble('data', ['model1.pth', 'model2.pth', 'model3.pth'], idx=0)