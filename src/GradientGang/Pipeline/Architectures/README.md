# Architectures Module

![Architecture Diagram](../../../../Deliverables/UML/PIPELINE.png)

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

### `WindowedModelWrapper`
Model-level windowing wrapper that applies sliding window segmentation internally during forward pass and aggregates window predictions for sample-level outputs. This ensures training/validation metrics match inference metrics, unlike data-level windowing which can create train/inference discrepancies.

**Configuration Parameters:**
- `base_model` (L.LightningModule): The base model to wrap (LightningAutoencoder or Direct)
- `window_size` (int): Size of each sliding window (default: 80)
- `stride` (int): Stride for sliding window (default: 40)
- `aggregation_method` (Literal): How to aggregate window predictions
  - `"avg_probs"`: Average softmax probabilities (recommended)
  - `"avg_logits"`: Average logits before softmax
  - `"majority_vote"`: Majority voting across windows
  - `"max_confidence"`: Pick prediction with highest confidence
- `window_loss_weight` (float): Weight for auxiliary window-level loss (0-1, default: 0.3)
  - 0.0: Only sample-level loss
  - 1.0: Only window-level loss
  - 0.3: 30% window loss, 70% sample loss (recommended)

**Key Features:**
- Receives full sequences (no windowing in DataLoader)
- Creates windows internally using efficient `torch.unfold`
- Aggregates window predictions to sample-level predictions
- Training/validation metrics match inference (no train/test gap)
- Respects base model's loss computation (reconstruction + classification for autoencoders)
- Supports both Direct and Autoencoder base models

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `base_model: L.LightningModule`<br>`window_size: int`<br>`stride: int`<br>`aggregation_method: str`<br>`window_loss_weight: float` | - | Initialize wrapper around base model. |
| `create_windows` | `time_series: Tensor` | `tuple[Tensor, int]` | Create sliding windows from full sequences using torch.unfold. Returns (windows, num_windows). |
| `forward` | `x: tuple` | `Tensor` | Forward pass with windowing. Returns sample_logits or (sample_logits, window_logits) during training. |
| `aggregate_predictions` | `window_logits: Tensor` | `Tensor` | Aggregate window-level predictions to sample-level using configured method. |
| `training_step` | `batch: tuple`<br>`batch_idx: int` | `Tensor` | Training step respecting base model's loss (reconstruction + classification for AE). |
| `validation_step` | `batch: tuple`<br>`batch_idx: int` | `float` | Validation step computing sample-level F1 and base model losses. |
| `predict_step` | `batch: tuple`<br>`batch_idx: int` | `tuple` | Prediction step returning sample-level predictions and probabilities. |
| `configure_optimizers` | - | `dict` | Configure AdamW optimizer using base model's learning rate. |

**Windowing Process:**
1. **Input**: Full sequences (batch, 34 features, 160 timesteps)
2. **Window Creation**: Sliding windows with torch.unfold
   - Fast path: Exact divisibility uses unfold directly
   - Slow path: Padding for non-divisible sequences
3. **Window Processing**: Each window processed by base model
4. **Aggregation**: Window predictions combined to sample-level
5. **Loss Computation**: 
   - Autoencoder: Reconstruction loss + classification loss (respects base model weighting)
   - Direct: Classification loss only
   - Window-level auxiliary loss (optional)

**Integration with FinalPipeline:**
```python
# FinalPipeline automatically wraps models when use_windowing=True
if use_windowing:
    model = WindowedModelWrapper(
        base_model=model,
        window_size=trial.suggest_int("window_size", 5, 40),
        stride=int(window_size * trial.suggest_categorical("stride_ratio", [0.25, 0.5, 0.75, 1.0])),
        aggregation_method=trial.suggest_categorical("aggregation_method", 
            ["avg_probs", "avg_logits", "majority_vote", "max_confidence"]),
        window_loss_weight=trial.suggest_float("window_loss_weight", 0.0, 0.5)
    )
```

**Advantages over Data-Level Windowing:**
- **Metric Consistency**: Training F1 = Validation F1 = Test F1
- **No Data Leakage**: Windows created per-sample, not across samples
- **Flexible Aggregation**: Multiple methods for combining window predictions
- **Base Model Preservation**: Respects base model's loss computation logic
- **Memory Efficient**: Windows created on-the-fly during forward pass

---

