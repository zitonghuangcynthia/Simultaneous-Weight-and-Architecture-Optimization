import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CreateDataset import CreateDataset
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def validation(autoencoder, activation_fn):   
  input_size = 3
  output_size = 1
  num_samples = 1000
  max_hidden_size = 7
  input_weight_lists = [9, 17, 25, 33]

  # Test
  all_median_percentage_errors = []
  autoencoder.eval()

  for it in range (50):
    with torch.no_grad():
      # Create dataset
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
            selected_rows = mask[torch.arange(mask.size(0)), row_indices].unsqueeze(1)  #[32, 1, 7]
            mask = selected_rows.repeat(1, 1000, 1)
            curr_result = activation_fn(curr_result) * mask
            prev_result = curr_result

        # Last layer
        sliced_matrix = decoded_matrices[:, :, (hidden_layer) * max_hidden_size : (hidden_layer) * max_hidden_size + 1]
        prediction = torch.matmul(curr_result, sliced_matrix)
        all_prediction.append(prediction)
      all_prediction = torch.stack(all_prediction).squeeze(3)

      all_output = torch.stack(outputs).squeeze(-1)
      all_output = all_output.unsqueeze(0)
      all_output = all_output.repeat(4, 1, 1)

      # Calculate loss in terms of MPE (Median Percentage Error)
      total_loss = 0
      abs_loss = torch.abs(all_prediction - all_output)
      PE = abs_loss / torch.abs(all_output)
      mid_loss, _ = torch.median(PE, dim=2)
      min_loss, _ = torch.min(mid_loss, dim=0)
      total_loss = torch.mean(min_loss)
      all_median_percentage_errors.append(total_loss.item())
  overall_median_percentage_error = np.mean(all_median_percentage_errors)

  return overall_median_percentage_error.item()
