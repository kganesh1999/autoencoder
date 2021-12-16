import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import sys
import os

# Function to split twenty test images for visualization
def train_test_split(actual_img, output_img, labels):
    temp = [0] * 10
    test_size = 20
    twenty_sample = []
    for i in range(len(actual_img)):
        if test_size != 0:
            if temp[labels[i]] < 2:
                temp[labels[i]] += 1
                twenty_sample.append([actual_img[i], output_img[i], labels[i]])
                test_size -= 1
    return twenty_sample


# Visualize Twenty Test Images
def visualize(visualize_data):
    #Visualizing test images in four batches
    batch_split = [0,5,10,15,20]
    set = 1
    for b in range(0,len(batch_split)-1):
        fig, axes = plt.subplots(5, 2)
        i = 0
        for v in visualize_data[batch_split[b]:batch_split[b+1]]:
            axes[i, 0].matshow(v[0].reshape(28, 28), cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 1].matshow(v[1].reshape(28, 28), cmap='gray')
            axes[i, 1].axis('off')
            i += 1
        fig.savefig("Image_set{}.png".format(set), format="PNG")
        set +=1
