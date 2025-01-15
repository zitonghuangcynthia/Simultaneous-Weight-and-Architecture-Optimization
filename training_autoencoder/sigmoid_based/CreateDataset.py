from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CustomDataset import CustomDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def CreateDataset(num_samples, input_size, output_size, max_hidden_size):
  # Create dataloader
  # List for generating dataset
  sparsity_rates_choices = [0.5, 0.7, 0.8, 0.9]
  num_hidden_layers_choices = [1, 2, 3, 4]

  all_loss = []
  avg_loss = []

  # Create dataset
  num_selected_sparsity = 8
  num_selected_hidden = 4
  num_hidden_layers = []
  sparsity_rates = []
  for i in range(num_selected_sparsity):
    random_sparse = random.choice(sparsity_rates_choices)
    sparsity_rates.append(random_sparse)

  for j in range(num_selected_hidden):
    num_hidden_layers.append(random.choice(num_hidden_layers_choices))


  dataset = CustomDataset(num_samples, input_size, output_size, num_hidden_layers, sparsity_rates, max_hidden_size)
  inputs = [data[0] for data in dataset]
  outputs = [data[1] for data in dataset]
  weights = [data[2] for data in dataset]
  lengths = [data[3] for data in dataset]

  return lengths, weights, inputs, outputs