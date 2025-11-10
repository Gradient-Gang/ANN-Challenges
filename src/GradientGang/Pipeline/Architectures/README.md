# Architectures Module

![Architecture Diagram](../../../../Deliverables/UML/UML_drawio.png)

## Description
The Architectures module provides a flexible framework for building deep learning models using PyTorch Lightning. It includes modular building blocks (Encoder, Decoder, FeedForward) and complete architectures (LightningAutoencoder, Direct) for multimodal classification tasks combining time series and global features. All architectures support parameter validation through the ParameterInterpreter utility and are configured via dictionaries for reproducible experiments.

---

## Architecture Classes

### `LightningAutoencoder`
Complete autoencoder architecture for multimodal data that reconstructs inputs while performing classification.

**Configuration Parameters:**
- `EncoderParams` (dict): Configuration for time series encoder
- `GlobalFFEncoderParams` (dict): Configuration for global features encoder
- `DecoderParams` (dict): Configuration for time series decoder
- `GlobalFFDecoderParams` (dict): Configuration for global features decoder
- `FeedForwardParams` (dict): Configuration for final classification layer
- `OutputDim` (int): Number of output classes
- `LearningRate` (float): Learning rate for optimizer
- `Patience` (int): Patience for learning rate scheduler
- `RegularizationWeight` (float): L2 regularization weight
- `num_input_channels` (int, optional): Input channels (default: 1)
- `base_channel_size` (int, optional): Base channel size (default: 64)
- `latent_dim` (int, optional): Latent dimension size (default: 128)
- `num_output_channels` (int, optional): Output channels (default: 1)
- `act_fn` (nn.Module, optional): Activation function (default: nn.GELU)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict` | - | Initialize the LightningAutoencoder with configuration parameters. |
| `forward` | `x: tuple` | `tuple` | Forward pass returning predictions and reconstructions. |
| `configure_optimizers` | - | `dict` | Configure AdamW optimizer and ReduceLROnPlateau scheduler. |
| `training_step` | `batch: tuple`<br>`batch_idx: int` | `Tensor` | Compute combined reconstruction and classification loss. |
| `validation_step` | `batch: tuple`<br>`batch_idx: int` | `float` | Compute and log F1 score and reconstruction loss. |
| `on_validation_epoch_end` | - | - | Reset F1 metric at epoch end. |
| `get_embeddings` | `x: Tensor` | `Tensor` | Extract latent embeddings from encoder. |

---

### `Direct`
Direct classification architecture without reconstruction, encoding inputs directly to predictions.

**Configuration Parameters:**
- `EncoderParams` (dict): Configuration for time series encoder
- `GlobalFFEncoderParams` (dict): Configuration for global features encoder
- `FeedForwardParams` (dict): Configuration for classification layers
- `OutputDim` (int): Number of output classes
- `LearningRate` (float): Learning rate for optimizer
- `Patience` (int): Patience for learning rate scheduler
- `RegularizationWeight` (float): L2 regularization weight
- `num_input_channels` (int, optional): Input channels (default: 1)
- `base_channel_size` (int, optional): Base channel size (default: 64)
- `latent_dim` (int, optional): Latent dimension size (default: 128)
- `act_fn` (nn.Module, optional): Activation function (default: nn.GELU)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict` | - | Initialize the Direct architecture with configuration parameters. |
| `forward` | `x: tuple` | `Tensor` | Forward pass returning class predictions. |
| `configure_optimizers` | - | `dict` | Configure AdamW optimizer and ReduceLROnPlateau scheduler. |
| `training_step` | `batch: tuple`<br>`batch_idx: int` | `Tensor` | Compute classification loss. |
| `validation_step` | `batch: tuple`<br>`batch_idx: int` | `float` | Compute and log F1 score. |
| `on_validation_epoch_end` | - | - | Reset F1 metric at epoch end. |

---

## Building Block Classes

### `Encoder`
Flexible encoder module supporting convolutional, recurrent (LSTM/GRU/RNN), and linear layers.

**Configuration Parameters:**
- `activation_function` (str): Activation function name ("ReLU", "GELU", "LeakyReLU")
- `layer_type` (list[dict]): List of layer configurations
- `num_input_channels` (int): Number of input channels
- `base_channel_size` (int): Base channel size for convolutions
- `latent_dim` (int): Output dimension of encoded representation
- `act_fn` (nn.Module): Activation function class

**Supported Layers:**
- Conv1d, Conv2d: 1D/2D convolutional layers
- Linear: Fully connected layers
- LSTM, GRU, RNN: Recurrent layers for sequence processing
- ConvTranspose2d: Transpose convolution
- Flatten: Flatten operation

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict`<br>`num_input_channels: int`<br>`base_channel_size: int`<br>`latent_dim: int`<br>`act_fn: nn.Module` | - | Initialize encoder with layer specifications. |
| `forward` | `x: Tensor` | `Tensor` | Encode input to latent representation. |

---

### `Decoder`
Flexible decoder module for reconstruction, supporting convolutional, recurrent, and linear layers.

**Configuration Parameters:**
- `activation_function` (str): Activation function name ("ReLU", "GELU", "LeakyReLU")
- `layer_type` (list[dict]): List of layer configurations
- `output_activation` (str, optional): Output activation ("Sigmoid", "Tanh")
- `latent_dim` (int): Input dimension from encoder
- `base_channel_size` (int): Base channel size for convolutions
- `num_output_channels` (int): Number of output channels
- `act_fn` (nn.Module): Activation function class

**Supported Layers:**
- ConvTranspose1d, ConvTranspose2d: Transpose convolutions for upsampling
- Linear: Fully connected layers
- LSTM, GRU, RNN: Recurrent layers for sequence reconstruction
- Conv1d, Conv2d: Standard convolutions
- Unflatten: Reshape operation

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict`<br>`latent_dim: int`<br>`base_channel_size: int`<br>`num_output_channels: int`<br>`act_fn: nn.Module` | - | Initialize decoder with layer specifications. |
| `forward` | `x: Tensor`<br>`seq_len: int` | `Tensor` | Decode latent representation to reconstructed output. |

---

### `FeedForward`
Fully connected feedforward network with support for normalization and dropout.

**Configuration Parameters:**
- `activation_function` (str): Activation function name ("ReLU", "GELU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "SELU")
- `layer_type` (list[dict]): List of layer configurations
- `output_activation` (str, optional): Output activation for final layer

**Supported Layers:**
- Linear: Fully connected layers
- Dropout: Dropout regularization
- BatchNorm1d: Batch normalization
- LayerNorm: Layer normalization

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict` | - | Initialize feedforward network with layer specifications. |
| `forward` | `x: Tensor` | `Tensor` | Forward pass through the network. |

---

