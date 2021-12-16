import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class AutoEncoder(nn.Module):
    def __init__(self, mode):
        super(AutoEncoder, self).__init__()
        if mode == 1:
            self.flatten = nn.Flatten(start_dim=1)
            self.encoder_fc = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True),
            )
            self.decoder_fc = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(True),
                nn.Linear(256, 784),
                nn.ReLU(True),
            )
        elif mode == 2:
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(8, 16, 3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim=1),
                nn.Linear(16*7*7,100),
                nn.Linear(100,10)
            )
            self.decoder_conv = nn.Sequential(
                nn.Linear(10,100),
                nn.Linear(100,16*7*7),
                nn.Unflatten(1,(16,7,7)),
                nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1),
                nn.ReLU(True)
            )
        else:
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else:
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)

    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def model_1(self, X):
        X = self.flatten(X)
        encoded = self.encoder_fc(X)
        reconstructed = self.decoder_fc(encoded)
        return reconstructed

    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        encoded = self.encoder_conv(X)
        # encoded = self.flatten()
        reconstructed = self.decoder_conv(encoded)
        return reconstructed
