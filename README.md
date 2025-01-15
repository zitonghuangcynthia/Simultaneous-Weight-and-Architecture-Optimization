# Simultaneous-Weight-and-Architecture-Optimization

- This is the code written in PyTorch for the paper SimultaneousWeight and Architecture Optimization for Neural Networks.
- For detailed explanation please refer to the following paper: Huang, Z., Montazerin, M., & Srivastava, A. (2024). Simultaneous Weight and Architecture Optimization for Neural Networks. arXiv preprint arXiv:2410.08339. [Link](https://arxiv.org/abs/2410.08339)

The repository is divided into two main parts:

# Train Autoencoder
   - This part focuses on training the autoencoder.
   - The purpose is to ensure that in the embedding space, two MLPs with similar functions are close to each other.
   - Code and resources related to this are located in the `training_autoencoder` folder.
   - **Usage:**
     ```bash
     python train_autoencoder.py --activation <activation_function>
     ```
     Replace `<activation_function>` with one of the following options:
     - `sigmoid`
     - `leakyrelu`
     - `linear`


# Train MLP
   - This part is responsible for training the MLP models.
   - Training consists of two stages:
     1. **Create Dataset:** Randomly generate an MLP with a specified activation and sparsity level, and generate enough input-output pairs to form a dataset.
     2. **Search MLP:** Train the MLP using the generated dataset.
   - Code and resources for this are found in the `training_MLP` folder.
   - **Usage:**
     1. Create Dataset:
        ```bash
        python create_dataset.py --activation <activation_function> --num_hidden_layer <num_layers> --sparsity <sparsity_level>
        ```
        - `<activation_function>`: Choose from `sigmoid`, `leakyrelu`, or `linear`.
        - `<num_layers>`: Number of hidden layers (1, 2, 3, or 4).
        - `<sparsity_level>`: A float between 0 and 1 indicating the sparsity level.
     2. Search MLP:
        ```bash
        python search_MLP.py --activation <activation_function>
        ```
        - `<activation_function>`: Must match the activation function used in dataset creation.

