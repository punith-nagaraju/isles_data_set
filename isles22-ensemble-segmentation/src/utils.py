import numpy as np
import matplotlib.pyplot as plt

def dice_score(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def plot_sample(x, y, pred, channel=4):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Input (center channel)")
    plt.imshow(x[channel].cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(y.cpu(), cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred.cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()