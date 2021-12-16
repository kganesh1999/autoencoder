from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from model import AutoEncoder
import argparse
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from utils import train_test_split, visualize

#Function to count trainable parameter in network
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    # Empty list to store losses
    losses = []

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        data = data.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)
        if FLAGS.mode == 1:
            data = data.view(-1, 784)
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output, data)

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

    epoch_loss = float(np.mean(losses))
    print('Epoch:{}/{} - Loss: {:.5f}'.format(epoch, FLAGS.num_epochs, epoch_loss))
    return epoch_loss


def test(model, device, test_loader, criterion, epoch):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()
    actual_imgs = []
    output_imgs = []
    labels = []
    losses = []
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(test_loader):
            data, target = batch_sample
            actual_imgs.append(data)
            data, target = data.to(device), target.to(device)
            if FLAGS.mode==1:
                data = data.view(-1, 784)

            # Predict for data by doing forward pass
            output = model(data)
            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            #

            # Remove NotImplementedError and assign correct loss function.
            # Compute loss based on same criterion as training

            loss = criterion(output, data)

            # Append loss to overall test loss
            losses.append(loss.item())
            output_imgs.append(output)
            labels.append(target)
    avg_loss = sum(losses)/len(losses)
    return avg_loss, actual_imgs, output_imgs, labels

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

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    filename = open(FLAGS.log_dir, 'w')
    sys.stdout = filename
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    # Initialize the model and send to device
    model = AutoEncoder(FLAGS.mode).to(device)

    # ======================================================================
    # Define loss function.
    criterion = nn.MSELoss()

    # ======================================================================
    # Define optimizer function.
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Create transformations to apply to each data sample
    # Can specify variations such as image flip, color flip, random crop, ...
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                              transform=transform)
    train_loader = DataLoader(dataset1, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size=1,
                             shuffle=False, num_workers=4)
    print('Number of trainable parameters =', count_parameters(model))
    epochs = []
    train_losses = []
    # Run training for n_epochs specified in config
    print("--------------------------Training the autoencoder--------------------------")
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train(model, device, train_loader,
                           optimizer, criterion, epoch, FLAGS.batch_size)
        epochs.append(epoch)
        train_losses.append(train_loss)

    print("--------------------------Training finished--------------------------")
    print('--------------------------Model Evaluation--------------------------')
    avg_test_loss, imgs, recons, labels = test(model, device, test_loader, criterion, epoch)
    print('Average test loss: {:.5f}'.format(avg_test_loss))
    print('--------------------------Visualizing from test samples--------------------------')
    visualize_20 = train_test_split(imgs, recons, labels)
    # To display test samples of size 20
    visualize(visualize_20)
    print('*****************************Check directory for visualization********************************')

if __name__ == '__main__':
    # Set parameters for Autoencoder
    parser = argparse.ArgumentParser('Auto Encoder.')
    parser.add_argument('--mode',
                        type=int, default=2,
                        help='Select mode between 1-2.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    run_main(FLAGS)

