from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CustomDataset import CustomDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def CreateDataset(num_samples, input_size, output_size, hidden_size, MLP_type, num_hidden_layer, sparsity):
  # Create dataloader
  # List for generating dataset
  sparsity_rates_choices = [sparsity]
  num_hidden_layers_choices = [num_hidden_layer]

  all_loss = []
  avg_loss = []

  # Create dataset
  num_selected_sparsity = 1
  num_selected_hidden = 1
  num_hidden_layers = []
  sparsity_rates = []
  for i in range(num_selected_sparsity):
    random_sparse = random.choice(sparsity_rates_choices)
    sparsity_rates.append(random_sparse)

  for j in range(num_selected_hidden):
    num_hidden_layers.append(random.choice(num_hidden_layers_choices))


  dataset = CustomDataset(num_samples, input_size, output_size, hidden_size, num_hidden_layers, sparsity_rates, MLP_type)
  inputs = [data[0] for data in dataset]
  outputs = [data[1] for data in dataset]
  weights = [data[2] for data in dataset]
  lengths = [data[3] for data in dataset]

  inputs = [torch.tensor(data, dtype=torch.float32).to(device) for data in inputs]
  outputs = [torch.tensor(data, dtype=torch.float32).to(device) for data in outputs]
  weights = [data.clone().to(device) for data in weights]
  lengths = [torch.tensor(data, dtype=torch.int64).to(device) for data in lengths]

  weights_list = []
  lengths_list = []

  for i in range (len(weights)):
    weight = weights[i]
    length = lengths[i]

    # feed the batch data into the autoencoder network and generate the new weight matrix
    if i % num_samples == 0:
      weights_list.append(weight)
      lengths_list.append(length)

  lengths_list = torch.tensor(lengths_list).to(device)
  return lengths_list, weights_list, inputs, outputs