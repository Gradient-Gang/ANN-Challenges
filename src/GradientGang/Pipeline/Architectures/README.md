# Architectures Module

This module contains neural network architecture implementations for various tasks including encoding, decoding, classification, and autoencoders.

---

## Encoder

### Description
A flexible encoder network that transforms input data into a lower-dimensional latent representation. Supports various layer types (Conv1d, Conv2d, Linear) and activation functions.

### Methods

#### `__init__(self, params: dict, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn=nn.GELU)`

**Parameters:**
- `params` (dict): Configuration dictionary with:
  - `activation_function` (str): Activation function name ("ReLU", "GELU", "LeakyReLU")
  - `layer_type` (list): List of layer configurations, each containing:
    - `name` (str): Layer type ("Conv2d", "Linear", "Conv1d", "ConvTranspose2d")
    - `params` (dict): Layer-specific parameters (in_channels, out_channels, kernel_size, etc.)
- `num_input_channels` (int): Number of input channels (e.g., 3 for RGB images)
- `base_channel_size` (int): Number of channels in first convolutional layer
- `latent_dim` (int): Dimensionality of latent representation
- `act_fn` (object): Default activation function class (default: nn.GELU)

#### `forward(self, x) -> torch.Tensor`
Forward pass through the encoder.

---

## Decoder

### Description
A flexible decoder network that reconstructs data from latent representations. Supports transposed convolutions for upsampling and various activation functions.

### Methods

#### `__init__(self, params: dict, latent_dim: int, base_channel_size: int, num_output_channels: int, act_fn=nn.GELU)`

**Parameters:**
- `params` (dict): Configuration dictionary with:
  - `activation_function` (str): Activation function for hidden layers
  - `output_activation` (str, optional): Activation for output layer ("Sigmoid", "Tanh")
  - `layer_type` (list): List of layer configurations
- `latent_dim` (int): Dimensionality of latent representation
- `base_channel_size` (int): Number of channels in last convolutional layer
- `num_output_channels` (int): Number of output channels (e.g., 3 for RGB)
- `act_fn` (object): Default activation function class (default: nn.GELU)

#### `forward(self, x) -> torch.Tensor`
Forward pass through the decoder.

---

## FeedForward

### Description
A flexible feedforward neural network supporting various layer types including Linear, Dropout, BatchNorm, and LayerNorm.

### Methods

#### `__init__(self, params: dict)`

**Parameters:**
- `params` (dict): Configuration dictionary with:
  - `activation_function` (str): Activation function ("ReLU", "GELU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "SELU")
  - `output_activation` (str, optional): Output layer activation
  - `layer_type` (list): List of layer configurations, each with:
    - `name` (str): Layer type ("Linear", "Dropout", "BatchNorm1d", "LayerNorm")
    - `params` (dict): Layer-specific parameters

#### `forward(self, x) -> torch.Tensor`
Forward pass through the network.

---

## LightningAutoencoder

### Description
A PyTorch Lightning module implementing an autoencoder with joint training for reconstruction and classification tasks. Combines encoder, decoder, and feedforward networks.

### Methods

#### `__init__(self, params: dict)`

**Parameters:**
- `params` (dict): Configuration with:
  - `EncoderParams` (dict): Encoder configuration
  - `DecoderParams` (dict): Decoder configuration
  - `FeedForwardParams` (dict): Classification head configuration
  - `OutputDim` (int): Number of output classes
  - `LearningRate` (float, optional): Learning rate (default: 0.001)
  - `Patience` (int, optional): Scheduler patience (default: 5)
  - `num_input_channels` (int, optional): Input channels (default: 1)
  - `base_channel_size` (int, optional): Base channel size (default: 64)
  - `latent_dim` (int, optional): Latent dimension (default: 128)
  - `num_output_channels` (int, optional): Output channels (default: 1)
  - `act_fn` (object, optional): Activation function (default: nn.GELU)

#### `forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]`
Forward pass returning (predictions, reconstructed_data).

#### `configure_optimizers()`
Configure AdamW optimizer with ReduceLROnPlateau scheduler.

#### `training_step(self, batch, batch_idx) -> torch.Tensor`
Training step computing reconstruction + classification loss.

#### `validation_step(self, batch, batch_idx) -> torch.Tensor`
Validation step computing F1 score.

---

## Direct

### Description
A direct classification network combining an encoder with a feedforward classifier, without reconstruction.

### Methods

#### `__init__(self, params: dict)`

**Parameters:**
- `params` (dict): Configuration with:
  - `EncoderParams` (dict): Encoder configuration
  - `FeedForwardParams` (dict): Classifier configuration
  - `OutputDim` (int): Number of output classes
  - `LearningRate` (float, optional): Learning rate
  - `Patience` (int, optional): Scheduler patience
  - Additional encoder parameters (num_input_channels, base_channel_size, etc.)

#### `forward(self, x) -> torch.Tensor`
Forward pass returning class predictions.

#### `configure_optimizers()`
Configure AdamW optimizer with ReduceLROnPlateau scheduler.

#### `training_step(self, batch, batch_idx) -> torch.Tensor`
Training step with cross-entropy loss.

#### `validation_step(self, batch, batch_idx) -> torch.Tensor`
Validation step computing F1 score.

---

## Example Configuration

```python
params = {
    "arch_type": "autoencoder_joint",
    "EncoderParams": {
        "activation_function": "GELU",
        "layer_type": [
            {
                "name": "Conv2d",
                "params": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1
                }
            }
        ]
    },
    "DecoderParams": {
        "activation_function": "GELU",
        "output_activation": "Sigmoid",
        "layer_type": [...]
    },
    "FeedForwardParams": {
        "activation_function": "ReLU",
        "layer_type": [...]
    },
    "OutputDim": 3,
    "LearningRate": 0.001
}
```
