import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from autoencoder import AutoEncoder
from MLP_train import MLP_train
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test import validation
import psutil
import logging
import os
import pickle
from pytorchtools import EarlyStopping
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse


# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

fixed_coeff = None

def training_plot(all_loss, avg_loss, num_plot, num_hidden_layer):
  """
    Plots the average loss during training.

    Args:
        all_loss (list): List of all loss values.
        avg_loss (list): List of average loss values.
        num_plot (int): Interval for sampling points to plot.
        num_hidden_layer (int): Number of hidden layers in the model.
  """
  avg_loss_np = [loss_item for loss_item in avg_loss]
  x_values = range(len(avg_loss_np))
  sampled_x_values = [i + 1 for i in range(0, len(x_values), num_plot)]
  sampled_avg_loss_np = avg_loss_np[::num_plot]
  
  plt.figure(figsize=(10, 5))
  plt.plot(sampled_x_values, sampled_avg_loss_np, marker='o', label='Average Loss')
  plt.title(f'Average Loss Record for Training {num_hidden_layer}')
  plt.xlabel('Number of Epoch')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.legend()  # Add legend to show labels
  plt.tight_layout()
  filename = f'Training_average_loss_plot_{num_hidden_layer}.png'
  plt.savefig(filename)
  print(f"Training average loss plot saved as {filename}")



def plot_validation(all_loss, num_plot, num_hidden_layer):
  """
    Plots the validation metrics over epochs.

    Args:
        all_loss (list): List of validation loss values.
        num_plot (int): Interval for sampling points to plot.
        num_hidden_layer (int): Number of hidden layers in the model.
  """
  all_loss_np = [loss_item for loss_item in all_loss]
  x_values = range(len(all_loss_np))
  sampled_x_values = [i + 1 for i in range(0, len(x_values), num_plot)]
  sampled_all_loss_np = all_loss_np[::num_plot]

  plt.figure(figsize=(10, 5))
  plt.plot(sampled_x_values, sampled_all_loss_np, marker='o', label='All MPE')
  plt.title(f'All Record for Validation {num_hidden_layer}')
  plt.xlabel('Number of Epoch')
  plt.ylabel('MPE')
  plt.grid(True)
  plt.legend()  # Add legend to show labels
  plt.tight_layout()
  filename = f'validation_plot_{num_hidden_layer}.png'
  plt.savefig(filename)
  print(f"Validation plot saved as {filename}")





def loss_fn(outputs, predictions, decoded_MLP, t, original_decoded, mask):
  """
    Computes the total loss for the MLP.

    Args:
        outputs (Tensor): Ground truth outputs.
        predictions (Tensor): Predicted outputs by the MLP.
        decoded_MLP (Tensor): Decoded MLP weights.
        t (Tensor): Trainable threshold parameter.
        original_decoded (Tensor): Original decoded MLP weights.
        mask (Tensor): Binary mask for sparsity enforcement.

    Returns:
        total_loss (float): Combined prediction and sparsity loss.
        sparsity (float): Sparsity penalty component of the loss.
  """
  pred_loss = ((outputs - predictions) ** 2).mean()
  global fixed_coeff
  if fixed_coeff is None:
      with torch.no_grad():
          fixed_coeff = (pred_loss / 7000).item()
  num_weights = 0

  smooth_threshold = 0.5 *  torch.sigmoid(10 * (torch.abs(original_decoded) - t))
  L1 = 0.1 * torch.abs(original_decoded)
  sparsity = fixed_coeff * (smooth_threshold + L1).sum()
  total_loss = pred_loss + sparsity

  return total_loss, sparsity



# Load autoencoder
input_size = 3
output_size = 1
num_samples = 1000
max_hidden_size = 7


input_height = max_hidden_size
input_weight_lists = [9, 17, 25, 33]
autoencoder = AutoEncoder(input_height, input_weight_lists)
# Please substitute this field into the giuven path or your own path
checkpoint = torch.load('/home/lies_mlp/workshop_code/checkpoints/sigmoid_based/9_epoch.pth')
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()

log_file = 'searched_MLP.txt'
memory_log_file = 'memory.log'

# Load the dataset (change it to the dataset you created)
with open('train.pkl', 'rb') as f:
  random_lengths_list, random_weights_list, inputs, outputs = pickle.load(f)

# Add argument parsing for activation type
parser = argparse.ArgumentParser(description="Train MLP with specified options.")
parser.add_argument("--activation", type=str, required=True, choices=['sigmoid', '', 'linear'], help="Activation type for MLP. Choose 'sigmoid', 'relu', or 'linear' depending on the model behavior you want to test.")
args = parser.parse_args()
activation = args.activation

# Run gradient descent for each of the decoder
num_hidden_layers_list = [1, 2, 3, 4]
for num_hidden_layer in num_hidden_layers_list:
  all_loss = []
  avg_loss = []
  all_valid = []

  z_size = 128
  z = torch.rand(1, z_size).to(device)
  z.requires_grad = True

  t = torch.tensor([0.0], device=device, requires_grad=True)
  best_loss = 1000
  best_loss = 1000

  # Define optimizer
  optimizer = optim.Adam([
      {'params': z, 'lr': 0.1},
      {'params': t, 'lr': 0.01}
      ])

  # For each decoder, train for 30000 iterations
  for epoch in range (30001):
    # Get decoded MLPs from embedding z
    concatenated_decode = autoencoder.reverse_fc(z)
    split_sizes = [encoder.flattened_size for encoder in autoencoder.encoders]
    concatenated_split = torch.split(concatenated_decode, split_sizes, dim=1)

    decoder = autoencoder.decoders[num_hidden_layer-1]
    decoded_MLP = decoder(concatenated_split[num_hidden_layer-1].reshape(128, 2, -1).unsqueeze(0), num_hidden_layer-1)
    mask = decoded_MLP[:, -num_hidden_layer:]
    ones_per_column = torch.sum(mask == 1, dim=0)
    decoded_MLP = decoded_MLP[:, :-num_hidden_layer]

    new_mask = torch.ones_like(decoded_MLP)
    new_mask[:2, :input_height] = 0
    new_mask[-2:, :input_height] = 0
    sub_total = 0

    for col_idx in range(mask.shape[1]):
        non_zero_rows = torch.nonzero(mask[:, col_idx], as_tuple=True)[0]
        for col in range(input_height):
            if col not in non_zero_rows:
                new_mask[:, sub_total + col] = 0
        sub_total = sub_total + input_height
        for row in range(decoded_MLP.shape[0]):
            if row not in non_zero_rows:
                new_mask[row, sub_total:min(decoded_MLP.shape[1], sub_total + input_height)] = 0

    # Get the final decoded_MLP using current embedding
    decoded_MLP = decoded_MLP * new_mask
    original_decoded = decoded_MLP

    # Apply soft max using our threshold t (trainable)
    soft_mask = torch.sigmoid(100 * (torch.abs(decoded_MLP) - t))
    decoded_MLP = decoded_MLP * soft_mask
  
    # Calculate predictions baseds on the activation function
    batch_prediction = MLP_train(input_size, max_hidden_size, output_size, decoded_MLP, num_hidden_layer, inputs, mask, activation)

    # Calculate loss
    loss, sparsity = loss_fn(outputs, batch_prediction, decoded_MLP, t, original_decoded, ones_per_column)
    if loss < best_loss:
      best_loss = loss
      best_mask = mask
      best_original = original_decoded
      best_t = t

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
      t.clamp_(min=0.0, max=3.0)


    all_loss.append(loss.item())
    curr_avg_loss = sum(all_loss) / len(all_loss)
    avg_loss.append(curr_avg_loss)

    if epoch % 200 == 0:
      print(f"Training iteration loss, {num_hidden_layer} hidden layers,  {epoch} epochs: ", all_loss[epoch])
      print(f"Training average loss, {num_hidden_layer} hidden layers, {epoch} epochs: ", avg_loss[epoch])
      print("t is: ", t)
      print("Number of 1s in each column:", ones_per_column)
      print()
    

    validation_MSE = validation(autoencoder, decoded_MLP, num_hidden_layer, mask, activation)
    if epoch % 400 == 0 and epoch > 0:
      training_plot(all_loss, avg_loss, (epoch // 100), num_hidden_layer)
      # Plot validation loss
      all_valid.append(validation_MSE)
      plot_validation(all_valid, 1, num_hidden_layer)
  

  # Store the matrix (with i hidden layer)
  with open(log_file, 'a') as f:
    f.write(f'[{num_hidden_layer}] Hidden Layers: {decoded_MLP}\n')
  print()
  # # In more details (the threshold, mask, loss, and original matrix before enforcing sparsity)
  # with open(log_file, 'a') as f:
  #   f.write(f'[{num_hidden_layer}] Hidden Layers: {best_original}, mask: {best_mask}, t is: {best_t}, Loss: {best_loss}\n')
    
  # Finish all
  print('All hidden layer optimizations completed.')



      

