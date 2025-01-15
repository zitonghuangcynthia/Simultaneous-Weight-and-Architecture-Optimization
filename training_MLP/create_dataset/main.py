import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CreateDataset import CreateDataset
from torch.utils.data import DataLoader
import psutil
import logging
import os
import pickle
import argparse


def save_dataset(MLP_type, num_hidden_layer, sparsity):
  """
    Generates and saves datasets based on input parameters.

    Args:
        MLP_type (str): Type of MLP (e.g., 'sigmoid', 'relu').
        num_hidden_layer (int): Number of hidden layers.
        sparsity (float): Sparsity level of the dataset.
  """

    
  input_size = 3
  output_size = 1
  hidden_size = 5
  num_samples = 100000

  # input and output has 1000 respectively
  random_lengths_list, random_weights_list, inputs, outputs = CreateDataset(num_samples, input_size, output_size, hidden_size, MLP_type, num_hidden_layer, sparsity)
  inputs = torch.stack(inputs)
  outputs = torch.stack(outputs)

  # Save the data
  base_filename = f"{MLP_type}_{num_hidden_layer}_hidden_{sparsity:.1f}"
  train_filename = f"{base_filename}_train.pkl"
  valid_filename = f"{base_filename}_valid.pkl"
  test_filename = f"{base_filename}_test.pkl"

  with open(train_filename, 'wb') as f:
    pickle.dump((random_lengths_list, random_weights_list, inputs[0:50000], outputs[0:50000]), f)

  with open(valid_filename, 'wb') as f:
    pickle.dump((random_lengths_list, random_weights_list, inputs[50001:80001], outputs[50001:80001]), f)

  with open(test_filename, 'wb') as f:
    pickle.dump((random_lengths_list, random_weights_list, inputs[80001:100001], outputs[80001:100001]), f)




if __name__ == "__main__":
    # Get parameters
    parser = argparse.ArgumentParser(description="Generate and save MLP datasets.")
    parser.add_argument("--MLP_type", type=str, required=True, help="Type of MLP (sigmoid, leakyrelu, linear).")
    parser.add_argument("--num_hidden_layer", type=int, required=True, help="Number of hidden layers.")
    parser.add_argument("--sparsity", type=float, required=True, help="Sparsity level of the dataset (e.g., 0.2).")
    args = parser.parse_args()

    # Validation parameters
    valid_mlp_types = {"sigmoid", "leakyrelu", "linear"}
    if args.MLP_type.lower() not in valid_mlp_types:
        raise ValueError(f"Invalid MLP_type '{args.MLP_type}'. Supported values are {valid_mlp_types}.")
    valid_hidden_layers = {1, 2, 3, 4}
    if args.num_hidden_layer not in valid_hidden_layers:
        raise ValueError(f"Invalid num_hidden_layer '{args.num_hidden_layer}'. Supported values are {valid_hidden_layers}.")
    if not (0.0 <= args.sparsity <= 1.0):
        raise ValueError(f"Invalid sparsity '{args.sparsity}'. It must be between 0.0 and 1.0 inclusive.")


    # Generate and save datasets
    save_dataset(args.MLP_type, args.num_hidden_layer, args.sparsity)
