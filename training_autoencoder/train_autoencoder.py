import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CreateDataset import CreateDataset
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
from test import validation
import matplotlib.pyplot as plt
import psutil
import logging
import os
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
import argparse

# Training plot within every epoch, measured in Mean Squared Error (MSE)
def training_plot(all_loss, avg_loss, num_plot, epoch):
  """
    Plots average loss during training.

    Args:
        all_loss (list): List of all loss values.
        avg_loss (list): List of average loss values.
        num_plot (int): Interval for sampling points to plot.
        epoch (int): Current epoch.
  """
  # Sample values
  avg_loss_np = [loss_item for loss_item in avg_loss]
  x_values = range(len(avg_loss_np))
  sampled_x_values = [i + 1 for i in range(0, len(x_values), num_plot)]
  sampled_avg_loss_np = avg_loss_np[::num_plot]
  
  # Plot all_loss as a line graph
  plt.figure(figsize=(10, 5))
  plt.plot(sampled_x_values, sampled_avg_loss_np, marker='o', label='Average Loss')
  plt.title(f'Average Loss Record for Training Epoch {epoch}')
  plt.xlabel('Number of Batches')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.legend() 
  plt.tight_layout()
  filename = f'Training_average_loss_plot_{epoch}.png'
  plt.savefig(filename)
  print(f"Training average loss plot saved as {filename}")


# Training loss plot after each epoch, measured in Mean Squared Error (MSE)
def overall_training_plot(training_avg_loss, save_path='all_training_loss.png'):
  """
    Plots training loss across epochs.

    Args:
        training_avg_loss (list): Average loss for each epoch.
        save_path (str): Path to save the plot.
  """
  epochs = range(1, 1 + len(training_avg_loss))  
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, training_avg_loss, marker='o', linestyle='-', label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.title('Training Loss vs. Epochs')
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()
  print(f"Training loss plot saved to {save_path}")


# Validation loss plot after each epoch, measured in Median Percentage Error (MPE)
def plot_validation(all_loss, num_plot):
  """
    Plots validation loss over time.

    Args:
        all_loss (list): List of validation loss values.
        num_plot (int): Interval for sampling points to plot.
  """
  # Sample values
  all_loss_np = [loss_item for loss_item in all_loss]
  x_values = range(len(all_loss_np))
  sampled_x_values = [i + 1 for i in range(0, len(x_values), num_plot)]
  sampled_all_loss_np = all_loss_np[::num_plot]

  # Plot all_loss as a line graph
  plt.figure(figsize=(10, 5))
  plt.plot(sampled_x_values, sampled_all_loss_np, marker='o', label='All MPE')
  plt.title(f'All Record for Validation')
  plt.xlabel('Number of Batches')
  plt.ylabel('MPE')
  plt.grid(True)
  plt.legend()  # Add legend to show labels
  plt.tight_layout()
  filename = f'validation_plot.png'
  plt.savefig(filename)
  print(f"Validation plot saved as {filename}")


# Initialize weights for autoencoder for better training
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
            init.xavier_uniform_(module.weight) 
            if module.bias is not None:
                init.zeros_(module.bias)


# Activation function selector
def get_activation_fn(activation):
    """
    Returns the activation function based on the specified type.

    Args:
        activation (str): Type of activation ('sigmoid', 'leakyrelu', 'linear').

    Returns:
        function: Corresponding activation function.
    """
    if activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'leakyrelu':
        return lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    elif activation == 'linear':
        return lambda x: x
    else:
        raise ValueError(f"Unsupported activation type: {activation}")


# Define parameters for initialize autoencoder
input_size = 3
output_size = 1
num_samples = 1000
max_hidden_size = 7
input_height = max_hidden_size
input_weight_lists = [9, 17, 25, 33]
autoencoder = AutoEncoder(input_height, input_weight_lists)
initialize_weights(autoencoder)

# Define optimizer with adaptive learning rate
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# Argument parsing for activation type
parser = argparse.ArgumentParser(description="Train MLP with specified options.")
parser.add_argument("--activation", type=str, required=True, choices=['sigmoid', 'leakyrelu', 'linear'], help="Activation type for MLP. Choose 'sigmoid', 'leakyrelu', or 'linear' depending on the model behavior you want to test.")
parser.add_argument("--cuda", type=str, default="cuda:0", help="CUDA device to use (e.g., 'cuda:0', 'cuda:1'). Use 'cpu' to run on CPU.")
args = parser.parse_args()
activation_type = args.activation
activation_fn = get_activation_fn(activation_type)


# Select device for training (GPU if available, otherwise CPU)
device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
logging.basicConfig(filename='memory_usage.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')



# Training for 10 epoches
all_loss = []
avg_loss = []
all_valid = []
autoencoder.train()

for epoch in range (10):

  # Define seeds for training on same dataset for each epoch
  # Since the dataset is too large, hard to generate before training
  torch.manual_seed(45) 
  np.random.seed(45) 

  all_loss = []
  avg_loss = []

  for batch in range (150001):

    # Get the dataset and feed the MLPs into the autoencoder
    lengths_list, weights_list, inputs, outputs = CreateDataset(num_samples, input_size, output_size, max_hidden_size, activation_fn)
    inputs_tensor = torch.stack(inputs)

    all_decoded = []
    for num_input_matrix in range(len(weights_list)):
      preprocess_weight = weights_list[num_input_matrix][:, :(weights_list[num_input_matrix].size(1)- lengths_list[num_input_matrix]*max_hidden_size - lengths_list[num_input_matrix])].unsqueeze(0).unsqueeze(0)
      decode = autoencoder(preprocess_weight)
      all_decoded.append(decode)
    new_all_decoded = [[] for _ in range(len(input_weight_lists))]
    for i in range(32):
      for j in range(len(input_weight_lists)):
        new_all_decoded[j].append(all_decoded[i][j])
    all_decode = []
    for element in new_all_decoded:
      new_element = torch.stack(element)
      all_decode.append(new_element)

    # Calculate the predicted values after feeding the input into the decoded MLPs
    all_prediction = []

    for hidden_layer, decoded_matrices in enumerate(all_decode):

      hidden_layer = hidden_layer + 1
      # calculate prediction and loss
      for i in range (hidden_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size,  decoded_matrices.size(2))
        sliced_matrix = decoded_matrices[:, :, start_index:end_index]
        if (i == 0):
          sliced_matrix = sliced_matrix[:,2:-2, :]
          mask = (sliced_matrix != 0).float()
          mask = mask[:, :1, :]
          mask = mask.repeat(1, inputs_tensor.size(1), 1)
          prev_result = torch.matmul(inputs_tensor, sliced_matrix)
          prev_result = activation_fn(prev_result) * mask
          curr_result = prev_result
        else:
          mask = (sliced_matrix != 0).float()
          curr_result = torch.matmul(prev_result, sliced_matrix)
          non_zero_mask = (mask != 0).any(dim=2).int() 
          row_indices = torch.argmax(non_zero_mask, dim=1)
          selected_rows = mask[torch.arange(mask.size(0)), row_indices].unsqueeze(1) 
          mask = selected_rows.repeat(1, 1000, 1)
          curr_result = activation_fn(curr_result) * mask
          prev_result = curr_result
      sliced_matrix = decoded_matrices[:, :, (hidden_layer) * max_hidden_size : (hidden_layer) * max_hidden_size + 1]
      prediction = torch.matmul(curr_result, sliced_matrix)
      all_prediction.append(prediction)
    all_prediction = torch.stack(all_prediction).squeeze(3)


    all_output = torch.stack(outputs).squeeze(-1)
    all_output = all_output.unsqueeze(0)
    all_output = all_output.repeat(4, 1, 1)

    # Calculate MSE loss between ground truth output values and predicted values
    total_loss = 0
    square_loss = torch.abs(all_prediction - all_output) ** 2
    mean_loss = torch.mean(square_loss, dim=2)
    min_loss, _ = torch.min(mean_loss, dim=0)
    total_loss = torch.mean(min_loss)

    # Zero the parameter gradients
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    all_loss.append(total_loss.item())
    curr_avg_loss = sum(all_loss) / len(all_loss)
    avg_loss.append(curr_avg_loss)


    if batch % 40 == 0:
      print(f"Training 7 max iteration loss, {epoch} epoch, {batch} batch: ", all_loss[batch])
      print(f"Training 7 max average loss, {epoch} epoch, {batch} batch: ", avg_loss[batch])
      print()


    if batch % 200 == 0 and batch > 0:
      training_plot(all_loss, avg_loss, (batch // 100), epoch)

  logging.info(f"Epoch {epoch}, Batch {batch}: GPU Memory - Allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, Cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
  logging.info(f"Epoch {epoch}, Batch {batch}: CPU Memory - Used: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

  torch.save(autoencoder.state_dict(), f"{epoch}_epoch.pth")
  scheduler.step()

  # Store the seed for main function (for the generation of training dataset)
  rng_state = torch.get_rng_state()
  np_rng_state = np.random.get_state()
  # Set a different seed for the generation of validation dataset
  torch.manual_seed(123)  
  np.random.seed(123)
  validation_error = validation(autoencoder, activation_fn)
  all_valid.append(validation_error)
  plot_validation(all_valid, 1)
  # Restore RNG state
  torch.set_rng_state(rng_state)
  np.random.set_state(np_rng_state)










