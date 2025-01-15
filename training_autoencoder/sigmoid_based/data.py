import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Generate random uniform input values between -1 and 1
def generate_input(input_size):
  x1_values = torch.linspace(-1, 1, steps=10, device=device)
  x2_values = torch.linspace(-1, 1, steps=10, device=device)
  x3_values = torch.linspace(-1, 1, steps=10, device=device)
  x1, x2, x3 = torch.meshgrid(x1_values, x2_values, x3_values, indexing='ij')
  inputs = torch.stack((x1.flatten(), x2.flatten(), x3.flatten()), dim=-1)
  return inputs

# Generate sparsity pattern
def generate_sparsity(input_size, output_size, num_hidden_layers, sparsity_rate, num_hidden_nodes):
  # Record the edges we remain between two layers, initialize with all 0
  sparsity_pattern = []
  if num_hidden_layers == 0:
    sparsity_pattern.append(torch.zeros((input_size, output_size), device=device))
  else:
    sparsity_pattern.append(torch.zeros((input_size, num_hidden_nodes[0]), device=device))
    for i in range(num_hidden_layers-1):
      sparsity_pattern.append(torch.zeros((num_hidden_nodes[i], num_hidden_nodes[i+1]), device=device))
    sparsity_pattern.append(torch.zeros((num_hidden_nodes[-1], output_size), device=device))

  # Ensuring there is at least one path from each input
  # Select the node in the first hidden layer
  first_hidden_node = torch.randint(num_hidden_nodes[0], (1,)).item()
  # Link all the input node with this selected node
  for input in range(input_size):
    sparsity_pattern[0][input][first_hidden_node] = 1
  # Iteratively select one node after the previous node (after the first_hidden_node)
  prev_node = first_hidden_node
  for layer in range(num_hidden_layers-1):
    node_index = torch.randint(num_hidden_nodes[layer+1], (1,)).item()
    sparsity_pattern[layer+1][prev_node][node_index] = 1
    prev_node = node_index
  # Between the last hidden layer and output layer
  if output_size > 1:
      output_index = torch.randint(output_size, (1,))
  else:
    output_index = 0

  sparsity_pattern[num_hidden_layers][prev_node][output_index] = 1

  # Prune from the remaining links
  curr_ones = sum((p == 1).sum().item() for p in sparsity_pattern)
  total_elements = sum(p.numel() for p in sparsity_pattern)
  target_ones = int(total_elements * (1-sparsity_rate))
  num_to_change = target_ones - curr_ones
 
  # Randomly choose (num_to_change) ones to be the edges, remain others as zeros
  all_indices = []
  for pattern_index in range(len(sparsity_pattern)):
    # Collect all indices which have edge weight as 0 now
    pattern = sparsity_pattern[pattern_index]
    zero_indices = torch.nonzero(pattern == 0, as_tuple=False)
    for zero_index in zero_indices:
      # Record layer and nodes information
      all_indices.append((pattern_index, zero_index[0], zero_index[1]))

  chosen_indices = []
  while len(chosen_indices) < num_to_change:
    index = torch.randint(len(all_indices), (1,), device=device).item()
    chosen_index = all_indices[index]
    pattern_index, start_node, end_node = chosen_index
    pattern = sparsity_pattern[pattern_index]
    if pattern[start_node, end_node] == 0:
      chosen_indices.append(chosen_index)
      pattern[start_node, end_node] = 1

  return sparsity_pattern


# Generate random weights between -10 and 10 for each remaining edges
def generate_weight(sparsity_model):
  weights = []
  for num, pattern in enumerate(sparsity_model):
    random_weight_pattern = torch.empty_like(pattern, dtype=torch.float32, device=device).uniform_(-10, 10)
    random_weight_pattern[pattern==0] = 0
    weights.append(random_weight_pattern)
  
  # We get the transpose of the weight matrix
  for layer, weight in enumerate(weights):
    weights[layer] = weights[layer].T

  return weights


# Calculate for the output values
class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, custom_weights, num_hidden_layer, pad_upper, pad_lower):
    super(MLP, self).__init__()
    self.num_hidden_layer = num_hidden_layer

    # Initialize layers
    self.fc_layers = nn.ModuleList()
    for i in range(num_hidden_layer):
      start_index = i * hidden_size
      end_index = min((i + 1) * hidden_size, custom_weights.size(1))
      layer_weights = custom_weights[:, start_index:end_index]
      if i == 0:
        layer_weights = layer_weights[pad_upper:-pad_lower, :]  # Apply slicing for the first layer

      self.fc_layers.append(nn.Linear(input_size, hidden_size, bias=False))
      self.fc_layers[-1].weight.data = layer_weights.T
      input_size = hidden_size

    # Final output layer
    self.output_layer = nn.Linear(hidden_size, output_size, bias=False)
    final_weights = custom_weights[:, num_hidden_layer * hidden_size:num_hidden_layer * hidden_size + 1]
    self.output_layer.weight.data = final_weights.T

  def forward(self, x, num_nodes):
    for i, layer in enumerate(self.fc_layers):
      x = torch.sigmoid(layer(x))
    x = self.output_layer(x)
    return x


# Preprocess the input MLPs
def data_preprocessing(input_data, num_nodes, num_hidden_layer):
  # Transpose each matrix
  transposed_data = [array.T for array in input_data]
  max_rows = 7
  max_cols = 7
  padded_matrices = []

  # Padding: if number of row of a matrix is smaller than the max_cols, add padding to that matrix
  first_layer_upper_padding = 0
  first_layer_lower_padding = 0
  for num, matrix in enumerate(transposed_data):
    if num == 0:
      padding_rows = max_rows - matrix.shape[0]
      upper_padding = padding_rows // 2
      lower_padding = padding_rows - upper_padding
      first_layer_upper_padding = upper_padding
      first_layer_lower_padding = lower_padding
      padded_matrix = F.pad(matrix, (0, max_cols - matrix.size(1), upper_padding, lower_padding))
    elif num == len(transposed_data)-1:
      padded_matrix = F.pad(matrix, (0, 0, 0, max_rows - matrix.size(0)))
    else:
      padded_matrix = F.pad(matrix, (0, max_cols - matrix.size(1), 0, max_rows - matrix.size(0)))

    padded_matrices.append(padded_matrix)

  # Concat the matrices into a single large matrix
  concatenated_matrix = torch.cat(padded_matrices, dim=1)
  mask = torch.zeros((max_rows, num_hidden_layer), device=device)
  for layer, num_node in enumerate(num_nodes):
    mask[:num_node, layer] = 1

  return concatenated_matrix, mask, first_layer_upper_padding, first_layer_lower_padding




# Randomly generate masks (which neuron is inactive) for each layer of MLPs
def generate_mask(num_hidden_layer):
  # Generate hidden layer size for each hidden layer (maximum is 10, minimum is 1)
  num_hidden_nodes = [random.randint(3, 7) for _ in range(num_hidden_layer)]  # Number of node in each hidden layer, maximum is 7, minimum is 3
  return num_hidden_nodes



# Generate and combine the whole dataset (input values, input MLPs, output values)
def generate_dataset(num_samples, input_size, output_size, num_hidden_layers, sparsity_rates, hidden_size):
  dataset = []

  for num_hidden_layer in num_hidden_layers:
    # Generate the neuron pattern (which node is active) in each layer
    num_hidden_nodes = generate_mask(num_hidden_layer)
    for sparsity_rate in sparsity_rates:
      # Generate sparsity pattern
      sparsity_model = generate_sparsity(input_size, output_size, num_hidden_layer, sparsity_rate, num_hidden_nodes)
      # Generate weights
      weights = generate_weight(sparsity_model)
      # Input weights Preprcessing
      weights_input, mask, pad_upper, pad_lower = data_preprocessing(weights, num_hidden_nodes, num_hidden_layer)
      input_MLP = torch.cat((weights_input, mask), dim=1)
      # Generate input
      input_data = generate_input(input_size)
      # Create MLP model
      mlp_model = MLP(input_size, hidden_size, output_size, weights_input.squeeze(0), num_hidden_layer, pad_upper, pad_lower)
      # Forward pass
      output = mlp_model(torch.Tensor(input_data), num_hidden_nodes)
      dataset.append((input_data, output, input_MLP))
  return dataset