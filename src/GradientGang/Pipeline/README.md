# Pipeline Module

## Description
The Pipeline module provides a unified framework for end-to-end deep learning workflows, orchestrating architecture building, hyperparameter optimization, training, and submission generation. It integrates modular components (DataLoader, Architectures, Optimizer, SubmissionGenerator, Utils) into a cohesive pipeline that enables reproducible experiments through configuration-based architecture construction. The module supports automated hyperparameter tuning with Optuna, flexible architecture selection, and seamless integration with PyTorch Lightning for efficient training workflows.

---

## Main Classes

### `Pipeline`
Central orchestration class that manages the complete machine learning workflow from data loading to model optimization.

**Configuration Parameters:**
- `dataset` (L.LightningDataModule): Training/validation data module
- `test` (L.LightningDataModule): Test data module
- `optimizer` (Optimizer): Hyperparameter optimizer instance
- `dict_config` (dict, optional): Configuration dictionary for architecture and training
- `path_config` (str, optional): Path to YAML configuration file

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `dataset: L.LightningDataModule`<br>`test: L.LightningDataModule`<br>`optimizer: Optimizer`<br>`dict_config: dict \| None`<br>`path_config: str \| None` | - | Initialize Pipeline with data modules and configuration. |
| `build_architecture` | `params: dict` | `L.LightningModule` | Build architecture based on configuration parameters. |
| `optimize` | - | `L.LightningModule` | Run hyperparameter optimization and return best model. |

**Supported Architecture Types:**
- `autoencoder_joint`: Joint autoencoder with shared latent space
- `autoencoder_split`: Split autoencoder with separate encoders
- `direct`: Direct classification without reconstruction

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Pipeline                            │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  DataLoader  │───▶│ Architecture │───▶│  Optimizer   │ │
│  │              │    │   Builder    │    │              │ │
│  │ - Train/Val  │    │ - Encoder    │    │ - Optuna     │ │
│  │ - Test Data  │    │ - Decoder    │    │ - TPE        │ │
│  │ - Batching   │    │ - FF Network │    │ - Callbacks  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         │                    │                    ▼         │
│         │                    │            ┌──────────────┐ │
│         │                    │            │   Training   │ │
│         │                    │            │  (Lightning) │ │
│         │                    │            └──────────────┘ │
│         │                    │                    │         │
│         │                    │                    ▼         │
│         ▼                    ▼            ┌──────────────┐ │
│  ┌──────────────────────────────────────▶│ Submission   │ │
│  │                                        │  Generator   │ │
│  │                                        └──────────────┘ │
│  │                                                          │
│  │         ┌──────────────────────────────────────┐        │
│  └────────▶│  ParameterInterpreter (Utils)        │        │
│            │  - Validation                        │        │
│            │  - Type Checking                     │        │
│            │  - String-to-Object Mapping          │        │
│            └──────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Components

### 1. **DataLoader**
Handles multimodal data loading (time series + global features).

**Key Features:**
- CSV-based data loading with automatic feature detection
- Train/validation/test splits
- Batch processing with parallel workers
- PyTorch Lightning DataModule integration

**Learn More:** [DataLoader README](./DataLoader/README.md)

---

### 2. **Architectures**
Modular building blocks and complete models for classification.

**Available Architectures:**
- **LightningAutoencoder**: Reconstruction + classification
- **Direct**: End-to-end classification
- **Encoder**: Flexible encoding (Conv, LSTM, GRU, Linear)
- **Decoder**: Flexible decoding for reconstruction
- **FeedForward**: Fully connected networks

**Learn More:** [Architectures README](./Architectures/README.md)

---

### 3. **Optimizer**
Automated hyperparameter optimization using Optuna.

**Key Features:**
- TPE (Tree-structured Parzen Estimator) sampling
- Support for categorical, float, and integer parameters
- Early stopping and model checkpointing
- F1 score optimization

**Learn More:** [Optimizer README](./Optimizer/README.md)

---

### 4. **SubmissionGenerator**
Automated CSV submission file creation for competitions.

**Key Features:**
- Batch prediction generation
- Custom label mapping
- Zero-padded sample indices
- Integration with trained models

**Learn More:** [SubmissionGenerator README](./SubmissionGenerator/README.md)

---

### 5. **Utils**
Parameter validation and interpretation utilities.

**Key Features:**
- Configuration validation with type checking
- Recursive nested parameter validation
- String-to-object mapping
- Error messages with context

**Learn More:** [Utils README](./Utils/README.md)

---

## Example Usage

### Basic Pipeline with Configuration File

```python
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer
import pytorch_lightning as L

# 1. Setup data modules
train_data_params = {
    "data_dir": "dataset/PirateProcessed",
    "train_file_name": "pirate_pain_train.csv",
    "train_file_name_labels": "pirate_pain_train_labels.csv",
    "test_file_name": "pirate_pain_test.csv",
    "batch_size": 32,
    "num_workers": 4,
    "val_split": 0.2
}

train_data = DataModule(train_data_params)
test_data = DataModule(train_data_params)

# 2. Create optimizer
optimizer = OptunaOptimizer()

# 3. Initialize pipeline with YAML config
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=optimizer,
    path_config="configs/model_config.yaml"
)

# 4. Run optimization and get best model
best_model = pipeline.optimize()

print(f"Best model: {best_model}")
```

---

### Pipeline with Dictionary Configuration

```python
# Define configuration as dictionary
config = {
    # Architecture type selection
    "arch_type": "direct",
    
    # Encoder configuration
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
                    "in_features": 3840,
                    "out_features": 128,
                    "bias": True
                }
            }
        ]
    },
    
    # Global features encoder
    "GlobalFFEncoderParams": {
        "activation_function": "ReLU",
        "layer_type": [
            {
                "name": "Linear",
                "params": {
                    "in_features": 2,
                    "out_features": 32,
                    "bias": True
                }
            }
        ]
    },
    
    # Classification head
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
            },
            {
                "name": "Dropout",
                "params": {
                    "p": 0.3,
                    "inplace": False
                }
            }
        ]
    },
    
    # Training parameters
    "OutputDim": 3,
    "LearningRate": 0.001,
    "Patience": 10,
    "RegularizationWeight": 0.0001,
    "latent_dim": 128
}

# Create pipeline with dict config
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=optimizer,
    dict_config=config
)

# Build single architecture without optimization
model = pipeline.build_architecture(config)

# Train manually
trainer = L.Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model, train_data)
```

---

### Complete Workflow: Data → Train → Submission

```python
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer
from GradientGang.Pipeline.SubmissionGenerator import generate_submission
import pytorch_lightning as L

# 1. Configure data
data_config = {
    "data_dir": "dataset/PirateProcessed",
    "train_file_name": "pirate_pain_train.csv",
    "train_file_name_labels": "pirate_pain_train_labels.csv",
    "test_file_name": "pirate_pain_test.csv",
    "batch_size": 64,
    "num_workers": 8,
    "val_split": 0.15,
    "globalFeaturesColumns": ["isPirate", "isNotPirate"],
    "primaryKeyColumn": "sample_index"
}

train_data = DataModule(data_config)
test_data = DataModule(data_config)

# 2. Setup pipeline
optimizer = OptunaOptimizer()
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=optimizer,
    path_config="configs/autoencoder_config.yaml"
)

# 3. Optimize hyperparameters
print("Starting hyperparameter optimization...")
best_model = pipeline.optimize()

# 4. Setup test data
test_data.setup(stage="test")
test_loader = test_data.test_dataloader()

# 5. Generate submission
print("Generating submission file...")
submission_df = generate_submission(
    model=best_model,
    dataloader=test_loader,
    output_path="submissions/final_submission.csv"
)

print(f"✓ Submission created with {len(submission_df)} predictions")
```

---

### Hyperparameter Optimization Configuration

```python
# Configuration with hyperparameter search space
optimization_config = {
    "arch_type": "direct",
    
    # Hyperparameters to optimize
    "latent_dim": {
        "type": "int",
        "params": {
            "name": "latent_dim",
            "low": 64,
            "high": 256,
            "step": 64
        }
    },
    
    "LearningRate": {
        "type": "float",
        "params": {
            "name": "LearningRate",
            "low": 1e-5,
            "high": 1e-2,
            "log": True
        }
    },
    
    "RegularizationWeight": {
        "type": "float",
        "params": {
            "name": "RegularizationWeight",
            "low": 1e-6,
            "high": 1e-2,
            "log": True
        }
    },
    
    # Fixed architecture params
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
                    "dropout": 0.2
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
                    "in_features": 2,
                    "out_features": 32,
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
    "Patience": 10
}

# Run optimization
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=OptunaOptimizer(),
    dict_config=optimization_config
)

best_model = pipeline.optimize()
```

---

### Comparing Multiple Architectures

```python
# Test different architecture types
architecture_types = ["direct", "autoencoder_joint", "autoencoder_split"]
results = {}

for arch_type in architecture_types:
    print(f"\nTesting architecture: {arch_type}")
    
    # Load architecture-specific config
    config_path = f"configs/{arch_type}_config.yaml"
    
    # Create fresh pipeline
    pipeline = Pipeline(
        dataset=train_data,
        test=test_data,
        optimizer=OptunaOptimizer(),
        path_config=config_path
    )
    
    # Build and train
    model = pipeline.build_architecture(pipeline.config)
    trainer = L.Trainer(max_epochs=30, accelerator="gpu")
    trainer.fit(model, train_data)
    
    # Evaluate
    val_results = trainer.validate(model, train_data)
    results[arch_type] = val_results[0]['val_f1']
    
    print(f"{arch_type} validation F1: {results[arch_type]:.4f}")

# Select best architecture
best_arch = max(results, key=results.get)
print(f"\nBest architecture: {best_arch} (F1: {results[best_arch]:.4f})")
```

---

### Custom Architecture Builder

```python
def custom_build_architecture(params):
    """
    Custom architecture builder for Pipeline integration.
    
    Args:
        params: Configuration dictionary from Pipeline
        
    Returns:
        Configured Lightning module
    """
    from GradientGang.Pipeline.Architectures import Direct, LightningAutoencoder
    
    arch_type = params.get("arch_type", "direct")
    
    if arch_type == "direct":
        return Direct(params)
    elif arch_type.startswith("autoencoder"):
        return LightningAutoencoder(params)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")

# Use custom builder in Pipeline
class CustomPipeline(Pipeline):
    def build_architecture(self, params):
        return custom_build_architecture(params)

pipeline = CustomPipeline(
    dataset=train_data,
    test=test_data,
    optimizer=optimizer,
    dict_config=config
)
```

---

## Configuration File Format (YAML)

### Basic Configuration

```yaml
# Architecture type
arch_type: direct

# Encoder configuration for time series
EncoderParams:
  activation_function: GELU
  layer_type:
    - name: Conv1d
      params:
        in_channels: 30
        out_channels: 64
        kernel_size: 3
        stride: 1
        padding: 1
        bias: true
    - name: Linear
      params:
        in_features: 3840
        out_features: 128
        bias: true

# Encoder for global features
GlobalFFEncoderParams:
  activation_function: ReLU
  layer_type:
    - name: Linear
      params:
        in_features: 2
        out_features: 32
        bias: true

# Classification head
FeedForwardParams:
  activation_function: ReLU
  layer_type:
    - name: Linear
      params:
        in_features: 160
        out_features: 64
        bias: true
    - name: Dropout
      params:
        p: 0.3
        inplace: false

# Training parameters
OutputDim: 3
LearningRate: 0.001
Patience: 10
RegularizationWeight: 0.0001
latent_dim: 128
```

---

### Autoencoder Configuration

```yaml
arch_type: autoencoder_joint

# Encoder configuration
EncoderParams:
  activation_function: GELU
  layer_type:
    - name: LSTM
      params:
        input_size: 30
        hidden_size: 128
        num_layers: 2
        batch_first: true
        dropout: 0.2
        bidirectional: false

GlobalFFEncoderParams:
  activation_function: ReLU
  layer_type:
    - name: Linear
      params:
        in_features: 2
        out_features: 32
        bias: true

# Decoder configuration
DecoderParams:
  activation_function: GELU
  output_activation: Sigmoid
  layer_type:
    - name: Linear
      params:
        in_features: 128
        out_features: 3840
        bias: true

GlobalFFDecoderParams:
  activation_function: ReLU
  layer_type:
    - name: Linear
      params:
        in_features: 32
        out_features: 2
        bias: true

# Classification head
FeedForwardParams:
  activation_function: ReLU
  layer_type:
    - name: Linear
      params:
        in_features: 160
        out_features: 64
        bias: true

# Training parameters
OutputDim: 3
LearningRate: 0.001
Patience: 10
RegularizationWeight: 0.0001
latent_dim: 128
base_channel_size: 64
num_input_channels: 1
num_output_channels: 1
```

---

### Optimization Configuration

```yaml
arch_type: direct

# Hyperparameters to optimize
latent_dim:
  type: int
  params:
    name: latent_dim
    low: 64
    high: 256
    step: 64

LearningRate:
  type: float
  params:
    name: LearningRate
    low: 0.00001
    high: 0.01
    log: true

RegularizationWeight:
  type: float
  params:
    name: RegularizationWeight
    low: 0.000001
    high: 0.01
    log: true

dropout_rate:
  type: float
  params:
    name: dropout_rate
    low: 0.0
    high: 0.5
    step: 0.05

# Fixed architecture configuration
EncoderParams:
  activation_function: GELU
  layer_type:
    - name: Conv1d
      params:
        in_channels: 30
        out_channels: 64
        kernel_size: 3
        stride: 1
        padding: 1
        bias: true

# ... rest of fixed params
```

---

## Architecture Type Selection

### `direct`
**Use Case:** Standard classification without reconstruction

**Characteristics:**
- Fastest training (no decoder)
- Lower memory footprint
- Direct input → classification
- Best for supervised learning tasks

**Configuration:**
- Requires: `EncoderParams`, `GlobalFFEncoderParams`, `FeedForwardParams`
- Optional: `latent_dim`, `LearningRate`, `Patience`

---

### `autoencoder_joint`
**Use Case:** Reconstruction + classification with shared latent space

**Characteristics:**
- Learns meaningful representations
- Regularization through reconstruction
- Better feature learning
- Useful for semi-supervised learning

**Configuration:**
- Requires: `EncoderParams`, `GlobalFFEncoderParams`, `DecoderParams`, `GlobalFFDecoderParams`, `FeedForwardParams`
- Optional: `latent_dim`, `base_channel_size`, `num_input_channels`, `num_output_channels`

---

### `autoencoder_split`
**Use Case:** Separate autoencoders for different modalities

**Characteristics:**
- Independent reconstruction for time series and global features
- More flexible than joint autoencoder
- Can handle different feature scales
- Best for heterogeneous multimodal data

**Configuration:**
- Requires: Same as `autoencoder_joint`
- Additional: Separate loss weights for each modality

---

## Workflow Patterns

### Pattern 1: Quick Experimentation

```python
# Minimal setup for quick tests
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer

data = DataModule(data_params)
pipeline = Pipeline(data, data, OptunaOptimizer(), path_config="config.yaml")
model = pipeline.build_architecture(pipeline.config)

# Quick training
trainer = L.Trainer(max_epochs=10, fast_dev_run=False)
trainer.fit(model, data)
```

---

### Pattern 2: Production Pipeline

```python
# Full pipeline with validation and submission
pipeline = Pipeline(train_data, test_data, optimizer, path_config="prod_config.yaml")

# Optimize with extensive search
best_model = pipeline.optimize()

# Evaluate on validation
trainer = L.Trainer(accelerator="gpu")
val_results = trainer.validate(best_model, train_data)

# Generate submission
test_data.setup(stage="test")
submission_df = generate_submission(
    model=best_model,
    dataloader=test_data.test_dataloader(),
    output_path="final_submission.csv"
)
```

---

### Pattern 3: Ensemble Pipeline

```python
# Train multiple models for ensemble
models = []
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]

for config_path in configs:
    pipeline = Pipeline(train_data, test_data, optimizer, path_config=config_path)
    model = pipeline.build_architecture(pipeline.config)
    
    trainer = L.Trainer(max_epochs=50)
    trainer.fit(model, train_data)
    models.append(model)

# Ensemble predictions
test_data.setup(stage="test")
test_loader = test_data.test_dataloader()

all_predictions = []
for model in models:
    generator = SubmissionGenerator(model, test_loader)
    predictions = generator.generate_predictions()
    all_predictions.append(predictions)

# Majority voting
ensemble_predictions, _ = torch.mode(torch.stack(all_predictions), dim=0)

# Generate ensemble submission
generator = SubmissionGenerator(models[0], test_loader)
submission_df = generator.create_submission_file(
    "ensemble_submission.csv",
    predictions=ensemble_predictions
)
```

---

## Parameter Validation

### Automatic Validation
Pipeline uses `ParameterInterpreter` to validate configurations:

1. **Required Parameters**: `arch_type` must be present
2. **Architecture Type**: Must be in `["autoencoder_joint", "autoencoder_split", "direct"]`
3. **Architecture-Specific**: Each architecture requires specific parameter sets
4. **Type Checking**: Parameters validated for correct types

### Validation Errors

```python
# Missing arch_type
config = {"EncoderParams": {...}}  # Missing arch_type
pipeline = Pipeline(train_data, test_data, optimizer, dict_config=config)
# Raises: KeyError: "Required parameter 'arch_type' not found"

# Invalid arch_type
config = {"arch_type": "invalid"}
pipeline = Pipeline(train_data, test_data, optimizer, dict_config=config)
# Raises: ValueError: "arch_type must be one of [...]"

# Wrong type
config = {"arch_type": 123}  # Should be string
# Raises: TypeError: "Parameter 'arch_type' must be of type str"
```

---

## Integration with External Tools

### TensorBoard Logging

```python
from lightning.pytorch.loggers import TensorBoardLogger

# Add logger to trainer
logger = TensorBoardLogger("logs", name="pipeline_experiment")

# Use in pipeline training
trainer = L.Trainer(
    max_epochs=50,
    logger=logger,
    accelerator="gpu"
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)

# View in TensorBoard
# tensorboard --logdir=logs
```

---

### Weights & Biases Integration

```python
from lightning.pytorch.loggers import WandbLogger

# Initialize W&B logger
wandb_logger = WandbLogger(
    project="pirate-pain-classification",
    name="pipeline-experiment",
    config=pipeline.config
)

# Train with W&B logging
trainer = L.Trainer(
    max_epochs=50,
    logger=wandb_logger,
    accelerator="gpu"
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

### MLflow Tracking

```python
from lightning.pytorch.loggers import MLFlowLogger

# Setup MLflow logger
mlflow_logger = MLFlowLogger(
    experiment_name="pipeline-optimization",
    tracking_uri="http://localhost:5000"
)

# Log configuration
mlflow_logger.log_hyperparams(pipeline.config)

# Train and track
trainer = L.Trainer(logger=mlflow_logger)
model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

## Advanced Features

### Custom Callbacks

```python
from lightning.pytorch.callbacks import Callback

class CustomCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} completed")

# Use in pipeline
trainer = L.Trainer(
    max_epochs=50,
    callbacks=[CustomCallback()]
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

### Gradient Accumulation

```python
# Train with gradient accumulation for larger effective batch size
trainer = L.Trainer(
    max_epochs=50,
    accumulate_grad_batches=4,  # Effective batch size = 4 * data batch size
    accelerator="gpu"
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

### Mixed Precision Training

```python
# Use FP16 for faster training
trainer = L.Trainer(
    max_epochs=50,
    precision=16,  # or "16-mixed" for PyTorch 2.0+
    accelerator="gpu"
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

### Multi-GPU Training

```python
# Distributed data parallel training
trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=4,  # Use 4 GPUs
    strategy="ddp"
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

## Error Handling

### Configuration Errors

```python
try:
    pipeline = Pipeline(
        dataset=train_data,
        test=test_data,
        optimizer=optimizer,
        dict_config=None,  # Invalid: both None
        path_config=None
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Output: dict_config or path_config have to be assigned
```

---

### Architecture Building Errors

```python
try:
    config = {"arch_type": "unknown_architecture"}
    model = pipeline.build_architecture(config)
except ValueError as e:
    print(f"Architecture error: {e}")
    # Output: 'unknown_architecture' not found in interpretation dictionary
```

---

### Data Module Errors

```python
try:
    invalid_data = "not a data module"
    pipeline = Pipeline(
        dataset=invalid_data,
        test=test_data,
        optimizer=optimizer,
        path_config="config.yaml"
    )
except ValueError as e:
    print(f"Data module error: {e}")
    # Output: dataset and test must be L.LightningDataModule
```

---

## Performance Optimization

### DataLoader Tuning

```python
# Optimize data loading for faster training
data_params = {
    # ... other params
    "batch_size": 128,       # Larger batches for GPU
    "num_workers": 8,        # More parallel workers
    "pin_memory": True,      # Faster GPU transfer (automatic)
    "persistent_workers": True  # Keep workers alive (automatic)
}
```

---

### Model Compilation (PyTorch 2.0+)

```python
# Compile model for faster execution
model = pipeline.build_architecture(pipeline.config)
compiled_model = torch.compile(model)

trainer = L.Trainer(max_epochs=50)
trainer.fit(compiled_model, train_data)
```

---

### Checkpoint Strategies

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# Save top-k best models
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="pipeline-{epoch:02d}-{val_f1:.4f}",
    monitor="val_f1",
    mode="max",
    save_top_k=3
)

trainer = L.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback]
)

model = pipeline.build_architecture(pipeline.config)
trainer.fit(model, train_data)
```

---

## Troubleshooting

**Issue**: "dict_config or path_config have to be assigned"
- **Solution**: Provide exactly one of `dict_config` or `path_config` (not both, not neither)

**Issue**: "dataset and test must be L.LightningDataModule"
- **Solution**: Ensure both dataset and test are instances of `L.LightningDataModule`

**Issue**: "Required parameter 'arch_type' not found"
- **Solution**: Add `arch_type` to configuration with valid value

**Issue**: Architecture building fails with interpretation error
- **Solution**: Check `arch_type` is in `["autoencoder_joint", "autoencoder_split", "direct"]`

**Issue**: YAML config file not loading
- **Solution**: Verify YAML syntax and file path is correct

**Issue**: Optimization returns None
- **Solution**: Check optimizer configuration and ensure dataloader is properly setup

**Issue**: Model training crashes during optimization
- **Solution**: Verify architecture parameters match data dimensions

**Issue**: Submission generation fails
- **Solution**: Ensure test dataloader is setup with `stage="test"`

---

## Best Practices

1. **Configuration Management**: Use YAML files for reproducibility
2. **Validation**: Always validate configurations before training
3. **Checkpointing**: Save models during training for recovery
4. **Logging**: Use experiment tracking (TensorBoard, W&B, MLflow)
5. **Data Splits**: Use consistent train/val splits with fixed seeds
6. **Hyperparameter Search**: Start with coarse grid, then refine
7. **Architecture Selection**: Test multiple architecture types
8. **Ensemble Methods**: Combine predictions from multiple models
9. **Error Handling**: Wrap pipeline operations in try-except blocks
10. **Documentation**: Document configurations and results

---

## Notes

- Pipeline requires exactly one of `dict_config` or `path_config`
- Both `dataset` and `test` must be `L.LightningDataModule` instances
- Configuration must include `arch_type` parameter
- Architecture interpretation uses `ParameterInterpreter` for validation
- Optimization uses architecture builder function pattern
- YAML configs loaded with `yaml.safe_load()`
- Architecture types are case-sensitive
- All paths in configs should be relative to execution directory
- Pipeline supports nested configuration dictionaries
- Optimizer integration follows builder pattern for flexibility
- Compatible with all PyTorch Lightning features
- Thread-safe for parallel dataloader workers
- Supports distributed training strategies

---

## Future Enhancements

- Multiple architecture type support in single config
- Automatic architecture selection based on data
- Configuration templating and inheritance
- Built-in cross-validation support
- Automatic hyperparameter suggestion
- Pipeline resumption from checkpoints
- Configuration versioning and tracking
- Automatic data preprocessing integration
- Model ensemble configuration
- A/B testing support for architectures
- Automated feature engineering integration
- Real-time training monitoring dashboard
- Automatic learning rate finder
- Neural architecture search (NAS) integration
- AutoML capabilities

---

## Related Documentation

- **Architectures**: [./Architectures/README.md](./Architectures/README.md)
- **DataLoader**: [./DataLoader/README.md](./DataLoader/README.md)
- **Optimizer**: [./Optimizer/README.md](./Optimizer/README.md)
- **SubmissionGenerator**: [./SubmissionGenerator/README.md](./SubmissionGenerator/README.md)
- **Utils**: [./Utils/README.md](./Utils/README.md)

---

## Quick Reference

### Minimal Working Example

```python
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer

# Data
data = DataModule({"data_dir": "data", "train_file_name": "train.csv", 
                   "train_file_name_labels": "labels.csv", 
                   "test_file_name": "test.csv", 
                   "batch_size": 32, "num_workers": 4, "val_split": 0.2})

# Pipeline
pipeline = Pipeline(data, data, OptunaOptimizer(), path_config="config.yaml")

# Train
model = pipeline.build_architecture(pipeline.config)
trainer = L.Trainer(max_epochs=50)
trainer.fit(model, data)
```

### Configuration Template

```yaml
arch_type: direct
EncoderParams: {...}
GlobalFFEncoderParams: {...}
FeedForwardParams: {...}
OutputDim: 3
LearningRate: 0.001
Patience: 10
RegularizationWeight: 0.0001
latent_dim: 128
```
