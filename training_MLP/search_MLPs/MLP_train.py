import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def MLP_train(input_size, hidden_size, output_size, custom_weights, num_hidden_layer, input, mask, activation):
  """
    Computes MLP predictions with specified activation function.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden layer.
        output_size (int): Size of output features.
        custom_weights (Tensor): Weights for the MLP layers.
        num_hidden_layer (int): Number of hidden layers.
        input (Tensor): Input data.
        mask (Tensor): Mask for sparsity enforcement.
        activation (str): Activation function ('sigmoid', 'leakyrelu', 'linear').

    Returns:
        Tensor: MLP predictions.
  """

  num_cols = custom_weights.size(1)
  num_rows = custom_weights.size(0)
  num_slicing = num_hidden_layer

  # Define activation function based on input argument
  if activation == 'sigmoid':
      activation_fn = torch.sigmoid
  elif activation == 'leakyrelu':
      activation_fn = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)  
  elif activation == 'linear':
      activation_fn = lambda x: x 
  else:
      raise ValueError(f"Unsupported activation type: {activation}")

  for i in range (num_slicing):
    start_index = i * hidden_size
    end_index = min((i + 1) * hidden_size, num_cols)
    sliced_matrix = custom_weights[:, start_index:end_index]
    if (i == 0):
      # Remove boundary rows for the first layer
      sliced_matrix = sliced_matrix[2:-2, :]
      new_mask = mask[:, i]
      prev_result = torch.matmul(input, sliced_matrix)
      prev_result = activation_fn(prev_result) * new_mask
      curr_result = prev_result
    else:
      new_mask = mask[:, i]
      curr_result = torch.matmul(prev_result, sliced_matrix)
      curr_result = activation_fn(curr_result) * new_mask
      prev_result = curr_result
  
  # Last layer
  sliced_matrix = custom_weights[:, (num_slicing) * hidden_size : (num_slicing) * hidden_size + 1]
  curr_result = torch.matmul(prev_result, sliced_matrix)
  prev_result = curr_result

  return curr_result