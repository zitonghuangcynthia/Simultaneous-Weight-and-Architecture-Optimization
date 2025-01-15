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
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def loss_fn(outputs, predictions):
  """
    Validates loss calculation.

    Args:
        outputs: The ground truth outputs.
        predictions: The predictions using searched MLPs.

    Returns:
        float: Validation loss (Median percentage error).
  """
  percentage_errors = ((predictions - outputs) / outputs.clamp(min=1e-8)) * 100
  pred_loss = percentage_errors.median()
  return pred_loss


def validation(autoencoder, decoded_MLP, num_hidden_layer, mask, activation):   
  """
    Validates the model using a validation dataset and specified activation function.

    Args:
        autoencoder: The trained autoencoder model.
        decoded_MLP: The decoded MLP that we want to validate.
        num_hidden_layer (int): Number of hidden layers.
        mask: Binary mask for sparsity enforcement.
        activation (str): Activation function ('sigmoid', 'leakyrelu', 'linear').

    Returns:
        float: Validation loss.
  """
  autoencoder.eval()

  input_size = 3
  output_size = 1
  num_samples = 1000
  max_hidden_size = 7

  # Load the dataset
  with open('valid.pkl', 'rb') as f:
    random_lengths_list, random_weights_list, inputs, outputs = pickle.load(f)

  # Define activation function based on input argument
  if activation == 'sigmoid':
      activation_fn = torch.sigmoid
  elif activation == 'leakyrelu':
      activation_fn = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1)  
  elif activation == 'linear':
      activation_fn = lambda x: x 
  else:
      raise ValueError(f"Unsupported activation type: {activation}")


  # Calculate predictions
  batch_prediction = MLP_train(input_size, max_hidden_size, output_size, decoded_MLP, num_hidden_layer, inputs, mask, activation)

  # Calculate loss
  loss = loss_fn(outputs, batch_prediction)

  return loss.item()
