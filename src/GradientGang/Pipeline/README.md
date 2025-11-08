# Pipeline Module

## Description
The Pipeline module orchestrates the complete machine learning workflow by integrating datasets, architectures, and optimization strategies. It serves as the main entry point for building and training neural network models, supporting multiple architecture types including autoencoders and direct classification networks.

## Main Class: `Pipeline`

### Methods

#### `__init__(self, dataset, test, optimizer, dict_config=None, path_config=None)`
Initialize the pipeline with datasets, optimizer, and configuration.

**Parameters:**
- `dataset` (L.LightningDataModule): Training/validation data module
- `test` (L.LightningDataModule): Test data module
- `optimizer` (Optimizer): Optimizer instance for hyperparameter tuning
- `dict_config` (dict, optional): Configuration dictionary (mutually exclusive with path_config)
- `path_config` (str, optional): Path to YAML configuration file (mutually exclusive with dict_config)

**Raises:**
- `ValueError`: If both or neither of dict_config/path_config are provided
- `ValueError`: If dataset/test are not LightningDataModule instances

---

#### `build_architecture(self, params: dict) -> L.LightningModule`
Build a neural network architecture based on provided parameters.

**Parameters:**
- `params` (dict): Architecture configuration including:
  - `arch_type` (str): Type of architecture ("autoencoder_joint", "autoencoder_split", or "direct")
  - Additional architecture-specific parameters

**Returns:**
- `L.LightningModule`: Instantiated PyTorch Lightning model

**Raises:**
- `NotImplementedError`: If "direct" architecture type is selected (not yet implemented)

---

#### `optimize(self)`
Run optimization process to find best hyperparameters.

**Returns:**
- Optimization study results (format depends on optimizer implementation)

## Configuration Format

The configuration should be a dictionary or YAML file with the following structure:

```yaml
arch_type: "autoencoder_joint"  # or "autoencoder_split", "direct"
# Additional architecture-specific parameters...
```

## Example Usage

```python
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer

# Create data modules
train_data = DataModule(train_params)
test_data = DataModule(test_params)

# Create optimizer
optimizer = OptunaOptimizer()

# Initialize pipeline
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=optimizer,
    path_config="config.yaml"
)

# Run optimization
results = pipeline.optimize()
```
