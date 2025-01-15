import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset generation
# Generate random uniform input between -1 and 1
def generate_input(input_size):
  input_array = (np.random.uniform(-1, 1, input_size))
  return input_array



def generate_sparsity(input_size, hidden_size, output_size, num_hidden_layers, sparsity_rate):
  # Record the edges we remain between two layers, initialize with all 0
  sparsity_pattern = []
  if num_hidden_layers == 0:
    sparsity_pattern.append(np.zeros((input_size, output_size)))
  else:
    sparsity_pattern.append(np.zeros((input_size, hidden_size)))
    for _ in range(num_hidden_layers-1):
      sparsity_pattern.append(np.zeros((hidden_size, hidden_size)))
    sparsity_pattern.append(np.zeros((hidden_size, output_size)))

  # Ensuring there is at least one path from each input
  # Select the node in the first hidden layer
  first_hidden_node = np.random.randint(hidden_size-1)
  # Link all the input node with this selected node
  for input in range(input_size):
    sparsity_pattern[0][input][first_hidden_node] = 1
  # Iteratively select one node after the previous node (after the first_hidden_node)
  prev_node = first_hidden_node
  for layer in range(num_hidden_layers-1):
    node_index = np.random.randint(hidden_size-1)
    sparsity_pattern[layer+1][prev_node][node_index] = 1
    prev_node = node_index
  # Between the last hidden layer and output layer
  if output_size > 1:
      output_index = np.random.randint(output_size)
  else:
    output_index = 0
  sparsity_pattern[num_hidden_layers][prev_node][output_index] = 1


  # Prune from the remaining links
  curr_zeros = sum(np.count_nonzero(pattern == 0) for pattern in sparsity_pattern)
  curr_ones = sum(np.count_nonzero(pattern == 1) for pattern in sparsity_pattern)
  target_ones = int((sum(pattern.size for pattern in sparsity_pattern)) * (1-sparsity_rate))
  num_to_change = target_ones - curr_ones

  # Randomly choose (num_to_change) ones to be the edges, remain others as zeros
  all_indices = []
  for pattern_index in range(len(sparsity_pattern)):
    # Collect all indices which have edge weight as 0 now
    pattern = sparsity_pattern[pattern_index]
    zero_indices = np.argwhere(pattern == 0)
    for zero_index in zero_indices:
      # Record layer and nodes information
      all_indices.append((pattern_index, zero_index[0], zero_index[1]))

  #print(all_indices)

  chosen_indices = []
  while len(chosen_indices) < num_to_change:
    index = np.random.choice(len(all_indices))
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
  for pattern in sparsity_model:
    random_weight_pattern = np.random.uniform(-10, 10, pattern.shape)
    random_weight_pattern[pattern==0] = 0
    weights.append(random_weight_pattern)

  # We get the transpose of the weight matrix
  for layer, weight in enumerate(weights):
    weights[layer] = weights[layer].T

  return weights




class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, custom_weights, num_hidden_layer, MLP_type):
    super(MLP, self).__init__()
    if num_hidden_layer > 0:
      self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False)])
      self.fc_layers.extend([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_hidden_layer-1)])
      self.fc_layers.append(nn.Linear(hidden_size, output_size, bias=False))

      # Give the custom weight matrix into the linear transformations
      for i, weight in enumerate(custom_weights):
        self.fc_layers[i].weight.data = torch.Tensor(weight)
    else:
      self.fc_layers = nn.ModuleList([nn.Linear(input_size, output_size, bias=False)])
      self.fc_layers[0].weight.data = torch.Tensor(custom_weights[0])
    
    # Define activation
    self.MLP_type = MLP_type
    if self.MLP_type == 'sigmoid':
        self.activation = torch.sigmoid
    elif self.MLP_type == 'leakyrelu':
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)
    elif self.MLP_type == 'linear':
        self.activation = lambda x: x  # Identity function for linear
    else:
        raise ValueError(f"Unsupported MLP_type: {MLP_type}. Supported types are 'sigmoid', 'leakyrelu', 'linear'.")


  def forward(self, x):
      for layer in self.fc_layers[:-1]:
        x = self.activation(layer(x))
      x = self.fc_layers[-1](x)
      return x



def data_preprocessing(input_data):
  # Transpose each matrix
  transposed_data = [np.transpose(array) for array in input_data]

  # Calculate maximum number of row
  max_rows = max(matrix.shape[0] for matrix in transposed_data)
  padded_matrices = []

  # Padding: if number of row of a matrix is smaller than the max_cols, add padding to that matrix
  for matrix in transposed_data:
      if matrix.shape[0] < max_rows:
          padding_rows = max_rows - matrix.shape[0]
          upper_padding = padding_rows // 2
          lower_padding = padding_rows - upper_padding
          padded_matrix = np.pad(matrix, ((upper_padding, lower_padding), (0, 0)), mode='constant')
      else:
          padded_matrix = matrix
      padded_matrices.append(padded_matrix)

  # Concat the matrices into a single latge matrix
  concatenated_matrix = np.concatenate(padded_matrices, axis=1)
  tensor_input = torch.tensor(concatenated_matrix, dtype=torch.float32)
  return tensor_input






# Generate the dataset
def generate_dataset(num_samples, input_size, output_size, hidden_size, num_hidden_layers, sparsity_rates, MLP_type):
  dataset = []
  for num_hidden_layer in num_hidden_layers:
    for sparsity_rate in sparsity_rates:
      # Generate sparsity pattern
      sparsity_model = generate_sparsity(input_size, hidden_size, output_size, num_hidden_layer, sparsity_rate)
      # Generate weights
      weights = generate_weight(sparsity_model)
      # print("Before: ", weights)
      # Input weights Preprcessing
      weights_input = data_preprocessing(weights)
      # print("after: ", weights_input)
      for _ in range(num_samples):
        # Generate input
        input_data = generate_input(input_size)
        # Create MLP model
        mlp_model = MLP(input_size, hidden_size, output_size, weights, num_hidden_layer, MLP_type)
        # Forward pass
        output = mlp_model(torch.Tensor(input_data))
        # Add sample to dataset
        dataset.append((input_data, output.detach().numpy(), weights_input))
  return dataset
