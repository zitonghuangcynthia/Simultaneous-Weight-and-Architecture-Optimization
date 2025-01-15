# Generate dataset, define dataloader
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from data import generate_dataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, num_samples, input_size, output_size, hidden_size, num_hidden_layers, sparsity_rates, MLP_type):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.sparsity_rates = sparsity_rates
        self.MLP_type = MLP_type


        self.data = generate_dataset(num_samples, input_size, output_size, hidden_size, num_hidden_layers, sparsity_rates, MLP_type)

        # Calculate the maximum column size
        max_column_size = 21

        # Add padding and stack them together
        self.padded_data = []
        self.lengths = []
        for _, _, data in self.data:
            padding = max_column_size - data.size(1)
            pad = nn.ZeroPad2d((0, padding, 0, 0))  # Padding only along the columns
            self.padded_data.append(pad(data))
            self.lengths.append(padding // 5)  # Assuming padding along the second dimension


    def __len__(self):
        # return self.num_samples * len(num_hidden_layers) * len(sparsity_rates)
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data, weight_matrix = self.data[idx]
        weight_matrix_padding = self.padded_data[idx]
        length = self.lengths[idx]
        return input_data, output_data, weight_matrix_padding, length