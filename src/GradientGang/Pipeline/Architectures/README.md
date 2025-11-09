# Architectures Module

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

## Example Usage

### LightningAutoencoder Configuration

```python
from GradientGang.Pipeline.Architectures import LightningAutoencoder
import pytorch_lightning as L

# Configuration dictionary
autoencoder_params = {
    "EncoderParams": {
        "activation_function": "GELU",
        "layer_type": [
            {
                "name": "Conv1d",
                "params": {
                    "in_channels": 30,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "bias": True
                }
            },
            {
                "name": "Linear",
                "params": {
                    "in_features": 64 * 60,
                    "out_features": 128,
                    "bias": True
                }
            }
        ]
    },
    "GlobalFFEncoderParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 5,
                    "out_features": 32,
                    "bias": True
                }
            }
        ]
    },
    "DecoderParams": {
        "activation_function": "GELU",
        "output_activation": "Sigmoid",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 128,
                    "out_features": 64 * 60,
                    "bias": True
                }
            }
        ]
    },
    "GlobalFFDecoderParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 32,
                    "out_features": 5,
                    "bias": True
                }
            }
        ]
    },
    "FeedForwardParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 160,
                    "out_features": 64,
                    "bias": True
                }
            }
        ]
    },
    "OutputDim": 3,
    "LearningRate": 0.001,
    "Patience": 5,
    "RegularizationWeight": 0.0001,
    "latent_dim": 128
}

# Create model
model = LightningAutoencoder(autoencoder_params)

# Train with Lightning Trainer
trainer = L.Trainer(max_epochs=50)
trainer.fit(model, train_dataloader, val_dataloader)
```

---

### Direct Architecture Configuration

```python
from GradientGang.Pipeline.Architectures import Direct

# Configuration for direct classification
direct_params = {
    "EncoderParams": {
        "activation_function": "GELU",
        "layer_type": [
            {
                "name": "LSTM",
                "params": {
                    "input_size": 30,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "batch_first": True,
                    "dropout": 0.2,
                    "bidirectional": False
                }
            }
        ]
    },
    "GlobalFFEncoderParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 5,
                    "out_features": 32,
                    "bias": True
                }
            },
            {
                "name": "Dropout",
                "params": {
                    "p": 0.2,
                    "inplace": False
                }
            }
        ]
    },
    "FeedForwardParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 160,
                    "out_features": 128,
                    "bias": True
                }
            },
            {
                "name": "Dropout",
                "params": {
                    "p": 0.3,
                    "inplace": False
                }
            },
            {
                "name": "Linear",
                "params": {
                    "in_features": 128,
                    "out_features": 64,
                    "bias": True
                }
            }
        ]
    },
    "OutputDim": 3,
    "LearningRate": 0.001,
    "Patience": 10,
    "RegularizationWeight": 0.0001,
    "latent_dim": 128
}

# Create and train model
model = Direct(direct_params)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloader, val_dataloader)
```

---

### Using Individual Building Blocks

```python
import torch
import torch.nn as nn
from GradientGang.Pipeline.Architectures import Encoder, Decoder, FeedForward

# Encoder example
encoder_params = {
    "activation_function": "GELU",
    "layer_type": [
        {
            "name": "Conv1d",
            "params": {
                "in_channels": 30,
                "out_channels": 64,
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "bias": True
            }
        },
        {
            "name": "Linear",
            "params": {
                "in_features": 64 * 30,
                "out_features": 128,
                "bias": True
            }
        }
    ]
}

encoder = Encoder(encoder_params, num_input_channels=30, 
                  base_channel_size=64, latent_dim=128, act_fn=nn.GELU)

# FeedForward example
ff_params = {
    "activation_function": "ReLU",
    "output_activation": "Softmax",
    "layer_type": [
        {
            "name": "Linear",
            "params": {"in_features": 128, "out_features": 256, "bias": True}
        },
        {
            "name": "Dropout",
            "params": {"p": 0.5, "inplace": False}
        },
        {
            "name": "Linear",
            "params": {"in_features": 256, "out_features": 10, "bias": True}
        }
    ]
}

feedforward = FeedForward(ff_params)

# Forward pass
x = torch.randn(32, 30, 60)  # batch_size=32, features=30, time=60
encoded = encoder(x)
output = feedforward(encoded)
```

---

### LSTM Encoder/Decoder Configuration

```python
# LSTM Encoder for time series
lstm_encoder_params = {
    "activation_function": "ReLU",
    "layer_type": [
        {
            "name": "LSTM",
            "params": {
                "input_size": 30,
                "hidden_size": 64,
                "num_layers": 2,
                "batch_first": True,
                "dropout": 0.2,
                "bidirectional": True
            }
        },
        {
            "name": "Linear",
            "params": {
                "in_features": 128,  # 64 * 2 for bidirectional
                "out_features": 64,
                "bias": True
            }
        }
    ]
}

# LSTM Decoder for reconstruction
lstm_decoder_params = {
    "activation_function": "ReLU",
    "layer_type": [
        {
            "name": "Linear",
            "params": {
                "in_features": 64,
                "out_features": 128,
                "bias": True
            }
        },
        {
            "name": "LSTM",
            "params": {
                "input_size": 128,
                "hidden_size": 64,
                "num_layers": 2,
                "batch_first": True,
                "dropout": 0.2,
                "bidirectional": False
            }
        },
        {
            "name": "Linear",
            "params": {
                "in_features": 64,
                "out_features": 30,
                "bias": True
            }
        }
    ]
}
```

---

## Architecture Patterns

### 1. Autoencoder with Reconstruction
- Encodes time series and global features separately
- Decodes both modalities for reconstruction
- Combines encoded features for classification
- Loss = Reconstruction Loss + Classification Loss

### 2. Direct Classification
- Encodes time series and global features separately
- Combines encoded features directly for classification
- No reconstruction component
- Loss = Classification Loss only

### 3. Multimodal Feature Fusion
- Time series → Encoder → Latent representation
- Global features → FeedForward → Feature vector
- Concatenate latent + features → Final classifier

---

## Key Features

### Parameter Validation
All architectures use `ParameterInterpreter` to validate configuration dictionaries, ensuring:
- Required parameters are present
- Parameter types are correct
- Layer configurations are valid

### Flexible Layer Types
Support for diverse layer types:
- **Convolutional**: Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
- **Recurrent**: LSTM, GRU, RNN (with bidirectional support)
- **Linear**: Fully connected layers
- **Normalization**: BatchNorm1d, LayerNorm
- **Regularization**: Dropout
- **Activations**: ReLU, GELU, LeakyReLU, Sigmoid, Tanh, ELU, SELU

### Automatic Optimization
- AdamW optimizer with weight decay
- ReduceLROnPlateau scheduler
- F1 score tracking for multiclass classification
- Automatic metric logging for TensorBoard

### Recurrent Layer Support
- Automatic handling of LSTM/GRU/RNN outputs
- Sequence length preservation for decoders
- Last-timestep extraction for encoders
- Bidirectional RNN support

---

## Training Workflow

1. **Define Configuration**: Create parameter dictionary with all required fields
2. **Instantiate Model**: Pass configuration to architecture class
3. **Create DataLoaders**: Use PyTorch Lightning DataModule
4. **Configure Trainer**: Set up Lightning Trainer with callbacks
5. **Train**: Call `trainer.fit(model, train_loader, val_loader)`
6. **Evaluate**: Use `trainer.validate()` or `trainer.test()`
7. **Extract Embeddings**: Use `model.get_embeddings()` for visualization

---

## Loss Functions

### LightningAutoencoder
- **Reconstruction Loss**: MSE between input and reconstructed output
- **Classification Loss**: Cross-entropy with optional class weights
- **Total Loss**: Sum of both components

### Direct
- **Classification Loss**: Cross-entropy with optional class weights

---

## Metrics

- **F1 Score**: Macro-averaged F1 for multiclass classification
- **Validation Loss**: Logged automatically during training
- **Reconstruction Loss**: Logged separately for autoencoders

---

## Notes

- All architectures inherit from `pytorch_lightning.LightningModule`
- Models automatically handle GPU/CPU placement
- Configuration dictionaries can be saved as YAML files
- Parameter validation occurs at initialization
- F1 metric resets automatically after each validation epoch
- Learning rate scheduling monitors validation F1 score
- Weight decay (L2 regularization) applied via AdamW optimizer
- Supports both 2D (batch, features) and 3D (batch, features, time) inputs

---

## Advanced Topics

### Custom Activation Functions
```python
import torch.nn as nn

params = {
    # ... other params
    "act_fn": nn.LeakyReLU  # Pass custom activation class
}
```

### Class Weights for Imbalanced Data
Modify `training_step` to use custom class weights:
```python
class_weights = torch.tensor([1.0, 2.0, 3.0], device=device)  # Adjust for your data
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

### Extracting Latent Representations
```python
# For LightningAutoencoder
embeddings = model.get_embeddings(input_data)

# For Direct architecture
model.eval()
with torch.no_grad():
    encoded = model.encoder(time_series_data)
```

### Custom Learning Rate Schedules
Override `configure_optimizers()` in your subclass to implement custom schedules.

---

## Troubleshooting

**Issue**: Dimension mismatch errors
- **Solution**: Verify layer input/output dimensions match in sequence

**Issue**: LSTM output shape errors
- **Solution**: Ensure `batch_first=True` in LSTM configuration

**Issue**: Parameter validation fails
- **Solution**: Check all required parameters are present and correctly typed

**Issue**: NaN losses during training
- **Solution**: Reduce learning rate, add gradient clipping, check data normalization

**Issue**: Reconstruction quality poor
- **Solution**: Increase latent dimension, add more layers, adjust regularization weight
