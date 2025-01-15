import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random


# Build up encoder network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out



class Encoder(nn.Module):
  def __init__(self, input_width, input_height):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.input_width = input_width
        self.input_height = input_height
        # compute the flattened size after convolutions
        self.flattened_size = (((self.input_width) + 3) // 4) * (((self.input_height) + 3) // 4) * 128

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            nn.Flatten()
        )


  def forward(self, x):
    # If it is not for this encoder, then output 0
    if x.size(2) != self.input_height or x.size(3) != self.input_width:
      x =  torch.zeros((1, self.flattened_size), device=device)
      x = x.unsqueeze(0)
      return x

    # If it is for this encoder, then just go through the net
    x = self.net(x)

    return x, self.flattened_size
  


# Build up decoder network (with different hidden layer numbers)
class Decoder(nn.Module):
  def __init__(self, flattened_size, output_width, output_height):
        super(Decoder, self).__init__()
        self.output_width = output_width
        self.output_height = output_height
        self.flattened_size = flattened_size

        self.curr_features = self.output_width * self.output_height
        self.prev_features = (self.flattened_size // 128) * 16

        self.net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            ResidualBlock(64, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            ResidualBlock(32, 1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )

        self.last_layer = nn.Sequential(
          nn.Flatten(),
          nn.Linear(self.prev_features, 512),
          nn.LeakyReLU(negative_slope=0.01, inplace=False),
          nn.Linear(512, 256),
          nn.LeakyReLU(negative_slope=0.01, inplace=False),
          nn.Linear(256, self.curr_features)
        )

  def forward(self, x, i):
      # reshape the tensor to match shape before flattening
      x = self.net(x)

      # Change to desired shape
      x = self.last_layer(x)  #([1, 55])
      # Reshape the flattened matrix
      x = x.view(self.output_height, -1) #([5, 11])
      # Apply sigmoid to the second channel and binarize it
      x[:, -(i+1):] = torch.sigmoid(x[:, -(i+1):])
      x[:, -(i+1):] = (x[:, -(i+1):] > 0.5).int()

      return x


class AutoEncoder(nn.Module):
    def __init__(self, input_height, input_weight_lists):
        super(AutoEncoder, self).__init__()

        self.input_weight_lists = input_weight_lists
        self.input_height = input_height

        # Encode
        # Define encoders
        self.encoders = nn.ModuleList([
            Encoder(input_width, self.input_height).to(device)
            for input_width in self.input_weight_lists
        ])

        # Calculate the flatten size in total
        total_flattened_size = sum(encoder.flattened_size for encoder in self.encoders)

        # Concat all the 4 encoder results and do a final fc layer -- single embedding space
        self.linear_transform = nn.Sequential(
            nn.Linear(total_flattened_size, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        ).to(device)


        # Decode
        # Reverse of linear_transformation
        self.reverse_fc = nn.Sequential(
              nn.Linear(128, 256),
              nn.LeakyReLU(negative_slope=0.01, inplace=False),
              nn.Linear(256, 512),
              nn.LeakyReLU(negative_slope=0.01, inplace=False),
              nn.Linear(512, total_flattened_size),
        ).to(device)
        
        # Decoders
        self.decoders = nn.ModuleList([
          Decoder(self.encoders[i].flattened_size, input_width, self.input_height).to(device)
            for i, input_width in enumerate(self.input_weight_lists)
        ])

    def forward(self, x):
        # Encode
        # Encoder network
        encoded_outputs = []
        for encoder in self.encoders:
          encoded_outputs.append(encoder(x))
      
        # Concatenate the results
        concatenated_encoded = torch.cat([enc[0] for enc in encoded_outputs], dim=1)
        # Do linear transformation to make it into the embedding space
        z = self.linear_transform(concatenated_encoded)
        

        # Decode
        # Reverse of fc layer to change it to the concatenation of  encoded matrices
        concatenated_decode = self.reverse_fc(z)

        # Split based on the flatten size
        split_sizes = [encoder.flattened_size for encoder in self.encoders]
        concatenated_split = torch.split(concatenated_decode, split_sizes, dim=1)

        # Feed into four decoders respectively
        all_decoded_matrix = []
        for i, decoder in enumerate(self.decoders):
          input_decoder_matrix = concatenated_split[i].reshape(128, 2, -1).unsqueeze(0)
          decoded = decoder(input_decoder_matrix, i)
          mask = decoded[:, -(i+1):]
          decoded_matrix = decoded[:, :-(i+1)]
          all_masks.append(mask)

          new_mask = torch.ones_like(decoded_matrix)
          new_mask[:2, :self.input_height] = 0
          new_mask[-2:, :self.input_height] = 0
          sub_total = 0
          
          for col_idx in range(mask.shape[1]):
              # Get non-zero elements
              non_zero_rows = torch.nonzero(mask[:, col_idx], as_tuple=True)[0]
              # Update this layer (make all the other columns to be 0)
              for col in range(self.input_height):
                  if col not in non_zero_rows:
                      new_mask[:, sub_total + col] = 0
              sub_total = sub_total + self.input_height
              # Update next layer (make all the other rows to be 0)
              for row in range(decoded_matrix.shape[0]):
                  if row not in non_zero_rows:
                      new_mask[row, sub_total:min(decoded_matrix.shape[1], sub_total + self.input_height)] = 0

          decoded_matrix = decoded_matrix * new_mask
          all_decoded_matrix.append(decoded_matrix)

        return all_decoded_matrix, all_masks
