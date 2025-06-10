import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Down part
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature
        # Up part
        for feature in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose3d(in_channels, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv3D(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            # Pad if needed
            if x.shape != skip.shape:
                diff = [s1 - s2 for s1, s2 in zip(skip.shape[2:], x.shape[2:])]
                x = F.pad(x, [d//2 for d in sum(zip(diff[::-1], diff[::-1]), ())])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return torch.sigmoid(self.final_conv(x))

def get_unet(in_channels=2, out_channels=1):
    return UNet3D(in_channels=in_channels, out_channels=out_channels)