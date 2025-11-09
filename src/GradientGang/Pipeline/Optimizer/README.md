# Optimizer Module

## Description
The Optimizer module provides hyperparameter optimization capabilities for neural network architectures using Optuna. It offers a flexible framework for automated hyperparameter tuning with support for categorical, float, and integer parameter spaces. The module integrates seamlessly with PyTorch Lightning and includes early stopping, model checkpointing, and TPE (Tree-structured Parzen Estimator) sampling for efficient optimization. It's designed to work with the Pipeline for end-to-end architecture optimization workflows.

---

## Main Classes

### `Optimizer` (Abstract Base Class)
Base class defining the interface for optimization strategies.

**Abstract Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `optimize` | `architecture_builder: types.FunctionType` | `Any` | Optimize the architecture builder function with hyperparameter search. |

---

### `OptunaOptimizer`
Concrete implementation of hyperparameter optimization using Optuna's TPE sampler.

**Configuration Parameters:**
- `params` (dict): Dictionary defining parameter search space with type and value ranges
- `n_trials` (int | None): Number of optimization trials to run (optional)

**Parameter Types Supported:**
- `categ`: Categorical parameters (discrete choices)
- `float`: Float parameters (continuous ranges with optional log scale)
- `int`: Integer parameters (discrete numeric ranges)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | - | - | Initialize OptunaOptimizer instance. |
| `optimize` | `architecture_builder: types.FunctionType`<br>`params: dict`<br>`n_trials: int \| None` | `optuna.study.Study` | Run optimization study and return results. |
| `objective` | `trial: optuna.trial.BaseTrial` | `float` | Objective function evaluated for each trial. |
| `getParams` | `params: dict`<br>`trial: optuna.trial.BaseTrial` | `dict` | Extract and interpret parameters for current trial. |

---

## Example Usage

### Basic Hyperparameter Optimization

```python
from GradientGang.Pipeline.Optimizer import OptunaOptimizer
import lightning as L

# Define architecture builder
def build_model(params):
    """
    Architecture builder that receives optimized parameters.
    """
    from GradientGang.Pipeline.Architectures import Direct
    
    model = Direct(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        learning_rate=params['learning_rate'],
        dropout=params['dropout']
    )
    return model

# Define parameter search space
param_space = {
    'input_dim': {
        'type': 'int',
        'params': {'name': 'input_dim', 'low': 64, 'high': 256, 'step': 64}
    },
    'hidden_dim': {
        'type': 'int',
        'params': {'name': 'hidden_dim', 'low': 128, 'high': 512, 'step': 128}
    },
    'num_layers': {
        'type': 'int',
        'params': {'name': 'num_layers', 'low': 2, 'high': 6}
    },
    'learning_rate': {
        'type': 'float',
        'params': {'name': 'learning_rate', 'low': 1e-5, 'high': 1e-2, 'log': True}
    },
    'dropout': {
        'type': 'float',
        'params': {'name': 'dropout', 'low': 0.0, 'high': 0.5}
    }
}

# Create optimizer and run study
optimizer = OptunaOptimizer()
study = optimizer.optimize(
    architecture_builder=build_model,
    params=param_space,
    n_trials=50
)

# Access best parameters
print("Best hyperparameters:")
print(study.best_params)
print(f"Best validation F1 score: {study.best_value}")
```

---

### Categorical Parameter Optimization

```python
# Optimize discrete choices
param_space = {
    'activation': {
        'type': 'categ',
        'params': {
            'name': 'activation',
            'choices': ['relu', 'gelu', 'tanh', 'leaky_relu']
        }
    },
    'optimizer_type': {
        'type': 'categ',
        'params': {
            'name': 'optimizer_type',
            'choices': ['adam', 'adamw', 'sgd', 'rmsprop']
        }
    },
    'batch_size': {
        'type': 'categ',
        'params': {
            'name': 'batch_size',
            'choices': [16, 32, 64, 128]
        }
    }
}

optimizer = OptunaOptimizer()
study = optimizer.optimize(build_model, param_space, n_trials=30)
```

---

### Mixed Parameter Types

```python
# Combine categorical, integer, and float parameters
param_space = {
    # Categorical: architecture type
    'encoder_type': {
        'type': 'categ',
        'params': {
            'name': 'encoder_type',
            'choices': ['lstm', 'gru', 'transformer']
        }
    },
    
    # Integer: layer dimensions
    'hidden_dim': {
        'type': 'int',
        'params': {'name': 'hidden_dim', 'low': 64, 'high': 512, 'step': 64}
    },
    
    # Integer: number of layers
    'num_layers': {
        'type': 'int',
        'params': {'name': 'num_layers', 'low': 1, 'high': 5}
    },
    
    # Float: learning rate with log scale
    'learning_rate': {
        'type': 'float',
        'params': {'name': 'learning_rate', 'low': 1e-5, 'high': 1e-2, 'log': True}
    },
    
    # Float: regularization
    'weight_decay': {
        'type': 'float',
        'params': {'name': 'weight_decay', 'low': 0.0, 'high': 0.1}
    },
    
    # Float: dropout probability
    'dropout': {
        'type': 'float',
        'params': {'name': 'dropout', 'low': 0.0, 'high': 0.5, 'step': 0.05}
    }
}

def build_flexible_model(params):
    """Builder supporting multiple encoder types."""
    if params['encoder_type'] == 'lstm':
        from GradientGang.Pipeline.Architectures import LSTMEncoder
        return LSTMEncoder(**params)
    elif params['encoder_type'] == 'gru':
        from GradientGang.Pipeline.Architectures import GRUEncoder
        return GRUEncoder(**params)
    else:
        from GradientGang.Pipeline.Architectures import TransformerEncoder
        return TransformerEncoder(**params)

optimizer = OptunaOptimizer()
study = optimizer.optimize(build_flexible_model, param_space, n_trials=100)
```

---

### Integration with DataModule

```python
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Optimizer import OptunaOptimizer

# Setup data
data_params = {
    "data_dir": "dataset/PirateProcessed",
    "train_file_name": "pirate_pain_train.csv",
    "train_file_name_labels": "pirate_pain_train_labels.csv",
    "test_file_name": "pirate_pain_test.csv",
    "batch_size": 32,
    "num_workers": 4,
    "val_split": 0.2
}

data_module = DataModule(data_params)

# Architecture builder with data
def build_with_data(params):
    from GradientGang.Pipeline.Architectures import Direct
    
    model = Direct(
        data_module=data_module,
        **params
    )
    return model

# Optimize
param_space = {
    'hidden_dim': {
        'type': 'int',
        'params': {'name': 'hidden_dim', 'low': 128, 'high': 512}
    },
    'learning_rate': {
        'type': 'float',
        'params': {'name': 'learning_rate', 'low': 1e-5, 'high': 1e-2, 'log': True}
    }
}

optimizer = OptunaOptimizer()
study = optimizer.optimize(build_with_data, param_space, n_trials=50)
```

---

### Analyzing Optimization Results

```python
# Run optimization
study = optimizer.optimize(build_model, param_space, n_trials=100)

# Best trial information
print(f"Best trial: {study.best_trial.number}")
print(f"Best value (F1 score): {study.best_value}")
print(f"Best parameters: {study.best_params}")

# Get all trials
trials_df = study.trials_dataframe()
print(trials_df.head())

# Plot optimization history
import optuna.visualization as vis

# Optimization history
fig1 = vis.plot_optimization_history(study)
fig1.show()

# Parameter importance
fig2 = vis.plot_param_importances(study)
fig2.show()

# Parallel coordinate plot
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# Slice plot
fig4 = vis.plot_slice(study)
fig4.show()
```

---

### Resume Interrupted Optimization

```python
# Save study to database
study_name = "architecture_optimization"
storage = "sqlite:///optuna_study.db"

optimizer = OptunaOptimizer()

# First run
study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=0)
)

# Note: Current implementation creates new study each time
# To resume, modify optimize method to accept existing study:
# study.optimize(optimizer.objective, n_trials=50)
```

---

### Custom Architecture Builder

```python
def build_complex_architecture(params):
    """
    Complex architecture with encoder-decoder structure.
    """
    from GradientGang.Pipeline.Architectures import (
        Encoder, Decoder, LightningAutoencoder
    )
    
    # Build encoder
    encoder = Encoder(
        input_dim=params['input_dim'],
        hidden_dim=params['encoder_hidden_dim'],
        latent_dim=params['latent_dim'],
        num_layers=params['encoder_layers'],
        dropout=params['dropout']
    )
    
    # Build decoder
    decoder = Decoder(
        latent_dim=params['latent_dim'],
        hidden_dim=params['decoder_hidden_dim'],
        output_dim=params['output_dim'],
        num_layers=params['decoder_layers'],
        dropout=params['dropout']
    )
    
    # Create autoencoder
    model = LightningAutoencoder(
        encoder=encoder,
        decoder=decoder,
        learning_rate=params['learning_rate']
    )
    
    return model

# Define comprehensive parameter space
autoencoder_params = {
    'input_dim': {
        'type': 'int',
        'params': {'name': 'input_dim', 'low': 128, 'high': 512}
    },
    'encoder_hidden_dim': {
        'type': 'int',
        'params': {'name': 'encoder_hidden_dim', 'low': 64, 'high': 256}
    },
    'latent_dim': {
        'type': 'int',
        'params': {'name': 'latent_dim', 'low': 16, 'high': 128}
    },
    'decoder_hidden_dim': {
        'type': 'int',
        'params': {'name': 'decoder_hidden_dim', 'low': 64, 'high': 256}
    },
    'encoder_layers': {
        'type': 'int',
        'params': {'name': 'encoder_layers', 'low': 2, 'high': 5}
    },
    'decoder_layers': {
        'type': 'int',
        'params': {'name': 'decoder_layers', 'low': 2, 'high': 5}
    },
    'learning_rate': {
        'type': 'float',
        'params': {'name': 'learning_rate', 'low': 1e-5, 'high': 1e-2, 'log': True}
    },
    'dropout': {
        'type': 'float',
        'params': {'name': 'dropout', 'low': 0.0, 'high': 0.5}
    },
    'output_dim': {
        'type': 'categ',
        'params': {'name': 'output_dim', 'choices': [3, 5, 10]}
    }
}

optimizer = OptunaOptimizer()
study = optimizer.optimize(build_complex_architecture, autoencoder_params, n_trials=75)
```

---

## Parameter Configuration Format

### Required Structure
Each parameter in the `params` dictionary must have:
```python
{
    'parameter_name': {
        'type': str,      # One of: 'categ', 'float', 'int'
        'params': dict    # Type-specific parameters
    }
}
```

---

### Categorical Parameters

```python
{
    'param_name': {
        'type': 'categ',
        'params': {
            'name': str,           # Parameter name (must match key)
            'choices': list        # List of possible values
        }
    }
}
```

**Example:**
```python
{
    'activation': {
        'type': 'categ',
        'params': {
            'name': 'activation',
            'choices': ['relu', 'gelu', 'tanh']
        }
    }
}
```

---

### Float Parameters

```python
{
    'param_name': {
        'type': 'float',
        'params': {
            'name': str,           # Parameter name (must match key)
            'low': float,          # Lower bound (inclusive)
            'high': float,         # Upper bound (inclusive)
            'log': bool,           # Optional: use log scale (default: False)
            'step': float          # Optional: discretization step
        }
    }
}
```

**Example (continuous):**
```python
{
    'dropout': {
        'type': 'float',
        'params': {
            'name': 'dropout',
            'low': 0.0,
            'high': 0.5
        }
    }
}
```

**Example (log scale):**
```python
{
    'learning_rate': {
        'type': 'float',
        'params': {
            'name': 'learning_rate',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        }
    }
}
```

**Example (discrete):**
```python
{
    'weight_decay': {
        'type': 'float',
        'params': {
            'name': 'weight_decay',
            'low': 0.0,
            'high': 0.1,
            'step': 0.01
        }
    }
}
```

---

### Integer Parameters

```python
{
    'param_name': {
        'type': 'int',
        'params': {
            'name': str,           # Parameter name (must match key)
            'low': int,            # Lower bound (inclusive)
            'high': int,           # Upper bound (inclusive)
            'step': int,           # Optional: step size (default: 1)
            'log': bool            # Optional: use log scale (default: False)
        }
    }
}
```

**Example (basic):**
```python
{
    'num_layers': {
        'type': 'int',
        'params': {
            'name': 'num_layers',
            'low': 2,
            'high': 6
        }
    }
}
```

**Example (with step):**
```python
{
    'hidden_dim': {
        'type': 'int',
        'params': {
            'name': 'hidden_dim',
            'low': 64,
            'high': 512,
            'step': 64  # Only test 64, 128, 192, 256, 320, 384, 448, 512
        }
    }
}
```

---

## Architecture Builder Function

### Function Signature
```python
def build_architecture(params: dict) -> L.LightningModule:
    """
    Build and return a PyTorch Lightning model.
    
    Args:
        params: Dictionary of hyperparameters selected by optimizer
        
    Returns:
        Configured Lightning module ready for training
    """
    pass
```

### Requirements
1. **Input**: Dictionary with keys matching parameter names
2. **Output**: `L.LightningModule` instance
3. **DataModule**: Must be configured externally or passed via params
4. **State**: Should be stateless (no shared mutable state)

### Example Template
```python
def build_architecture(params: dict) -> L.LightningModule:
    # Import architecture
    from GradientGang.Pipeline.Architectures import Direct
    
    # Create model with optimized parameters
    model = Direct(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        output_dim=params['output_dim'],
        num_layers=params['num_layers'],
        learning_rate=params['learning_rate'],
        dropout=params['dropout'],
        activation=params['activation']
    )
    
    return model
```

---

## Optimization Process

### 1. Study Creation
- Creates Optuna study with TPE sampler
- Fixed seed (0) for reproducibility
- Direction: Maximize validation F1 score

### 2. Trial Execution
For each trial:
1. **Parameter Selection**: TPE sampler suggests parameter values
2. **Parameter Interpretation**: `getParams()` converts config to values
3. **Architecture Building**: `architecture_builder()` creates model
4. **Training**: PyTorch Lightning Trainer with callbacks
5. **Evaluation**: Returns best validation F1 score

### 3. Callbacks
- **EarlyStopping**: Monitors `val_f1`, patience=10, mode=max
- **ModelCheckpoint**: Saves best model based on `val_f1`

### 4. Result Collection
- Study object contains all trials
- Best parameters and score accessible
- Trial history for analysis

---

## TPE Sampler

### What is TPE?
Tree-structured Parzen Estimator is a Bayesian optimization algorithm that:
- Models promising and unpromising regions separately
- Adaptively focuses on promising hyperparameter regions
- More efficient than random or grid search

### Advantages
- **Sample Efficiency**: Fewer trials needed vs random search
- **Adaptive**: Learns from previous trials
- **Robust**: Works well with mixed parameter types
- **Parallelizable**: Supports distributed optimization

### Seed Configuration
- Fixed seed (0) ensures reproducibility
- Same parameter sequence across runs
- Deterministic trial ordering

---

## Integration with Lightning

### Training Configuration
```python
# Default trainer setup in objective function
trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stopping]
)
trainer.fit(arch)
```

### Customizing Trainer
To customize trainer settings, modify the architecture builder:

```python
def build_with_custom_trainer(params):
    model = create_model(params)
    
    # Custom trainer configuration
    model.trainer_kwargs = {
        'max_epochs': 100,
        'accelerator': 'gpu',
        'devices': 1,
        'precision': 16
    }
    
    return model
```

---

## Monitoring and Callbacks

### Early Stopping
```python
early_stopping = EarlyStopping(
    monitor="val_f1",    # Metric to monitor
    patience=10,         # Epochs without improvement
    mode="max"           # Maximize F1 score
)
```

**Customization:**
- Change `monitor` to track different metrics
- Adjust `patience` for longer/shorter training
- Use `mode="min"` for loss metrics

### Model Checkpointing
```python
checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",
    mode="max"
)
```

**Features:**
- Saves best model automatically
- Accessible via `checkpoint_callback.best_model_path`
- Score via `checkpoint_callback.best_model_score`

---

## Parameter Interpreter

### Role
`ParameterInterpreter` validates and interprets parameter configurations.

### Validation
Checks for required fields:
- `type`: Parameter type ('categ', 'float', 'int')
- `params`: Type-specific configuration dictionary

### Interpretation Mapping
```python
interpretation = {
    "categ": trial.suggest_categorical,
    "float": trial.suggest_float,
    "int": trial.suggest_int
}
```

Maps parameter types to Optuna suggestion methods.

---

## Advanced Usage

### Multi-Objective Optimization

```python
# Optimize multiple metrics simultaneously
# Note: Requires modification to current implementation

def multi_objective(trial):
    params = optimizer.getParams(param_space, trial)
    arch = build_architecture(params)
    
    # Train model
    trainer.fit(arch)
    
    # Return multiple objectives
    f1_score = checkpoint_callback.best_model_score
    model_size = count_parameters(arch)
    
    return f1_score, -model_size  # Maximize F1, minimize size

# Use multi-objective study
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=0)
)
```

---

### Pruning Unpromising Trials

```python
# Add pruning callback for early trial termination
# Requires architecture to report intermediate values

from lightning.pytorch.callbacks import Callback

class OptunaPruningCallback(Callback):
    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor
    
    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        
        self.trial.report(current_score, epoch)
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# Use in architecture builder
def build_with_pruning(params, trial):
    model = create_model(params)
    pruning_callback = OptunaPruningCallback(trial, "val_f1")
    # Add to trainer callbacks
    return model
```

---

### Conditional Parameters

```python
# Parameters that depend on other parameters
def build_conditional(params):
    if params['use_attention']:
        # Use attention heads only if attention is enabled
        num_heads = params['num_attention_heads']
    else:
        num_heads = 1
    
    model = create_model(
        use_attention=params['use_attention'],
        num_heads=num_heads,
        # ... other params
    )
    return model

# Define parameter space with optional parameters
param_space = {
    'use_attention': {
        'type': 'categ',
        'params': {'name': 'use_attention', 'choices': [True, False]}
    },
    'num_attention_heads': {
        'type': 'int',
        'params': {'name': 'num_attention_heads', 'low': 1, 'high': 8}
    }
}
```

---

## Performance Tips

1. **Start Small**: Begin with fewer trials (10-20) to validate setup
2. **Narrow Search Space**: Use domain knowledge to limit ranges
3. **Log Scale**: Use log scale for learning rates and regularization
4. **Step Sizes**: Use steps for integer parameters to reduce search space
5. **Warm Start**: Use previous study results as starting point
6. **Parallel Trials**: Run multiple trials in parallel (requires distributed setup)
7. **Monitor Progress**: Use Optuna dashboard for real-time monitoring

---

## Common Parameter Ranges

### Learning Rate
```python
'learning_rate': {
    'type': 'float',
    'params': {'name': 'learning_rate', 'low': 1e-5, 'high': 1e-2, 'log': True}
}
```

### Hidden Dimensions
```python
'hidden_dim': {
    'type': 'int',
    'params': {'name': 'hidden_dim', 'low': 64, 'high': 512, 'step': 64}
}
```

### Number of Layers
```python
'num_layers': {
    'type': 'int',
    'params': {'name': 'num_layers', 'low': 2, 'high': 6}
}
```

### Dropout
```python
'dropout': {
    'type': 'float',
    'params': {'name': 'dropout', 'low': 0.0, 'high': 0.5, 'step': 0.05}
}
```

### Batch Size
```python
'batch_size': {
    'type': 'categ',
    'params': {'name': 'batch_size', 'choices': [16, 32, 64, 128, 256]}
}
```

### Weight Decay
```python
'weight_decay': {
    'type': 'float',
    'params': {'name': 'weight_decay', 'low': 1e-6, 'high': 1e-2, 'log': True}
}
```

---

## Error Handling

### Parameter Validation
```python
# ParameterInterpreter checks for required fields
try:
    params = optimizer.getParams(param_space, trial)
except KeyError as e:
    print(f"Missing required parameter field: {e}")
```

### Architecture Builder Errors
```python
def robust_builder(params):
    try:
        model = create_model(params)
        return model
    except Exception as e:
        print(f"Error building architecture: {e}")
        # Return default model or raise
        raise
```

### Training Failures
- Early stopping prevents indefinite training
- Model checkpoint saves best result even if training fails
- Optuna handles trial exceptions gracefully

---

## Troubleshooting

**Issue**: "No attribute 'data_module' in architecture"
- **Solution**: Pass DataModule in architecture builder or ensure model has data_module configured

**Issue**: Optimization is very slow
- **Solution**: Reduce `n_trials`, simplify architecture, or use pruning

**Issue**: All trials produce similar results
- **Solution**: Widen parameter ranges or check if parameters actually affect model

**Issue**: Best score is NaN or invalid
- **Solution**: Check data preprocessing, loss function, and learning rate range

**Issue**: Study crashes during optimization
- **Solution**: Add error handling in architecture builder, validate parameters

**Issue**: Parameters not being applied to model
- **Solution**: Verify parameter names match exactly between config and builder

---

## Notes

- Optimization metric: Validation F1 score (higher is better)
- TPE sampler uses seed=0 for reproducibility
- Early stopping patience is fixed at 10 epochs
- Model checkpoint monitors validation F1
- Parameter names in config must match builder function expectations
- All parameters are passed as keyword arguments to builder
- Study object contains complete optimization history
- Best parameters accessible via `study.best_params`
- Supports distributed optimization with appropriate Optuna storage backend
- Compatible with all PyTorch Lightning training strategies
- Requires Lightning module with configured data module
- Callback configurations are currently fixed (requires code modification to change)

---

## Future Enhancements

- Multi-objective optimization support
- Custom callback configuration
- Distributed trial execution
- Resume from saved studies
- Custom metric monitoring
- Pruning unpromising trials
- Hyperparameter importance analysis
- Automatic parameter range suggestion
- Integration with experiment tracking (MLflow, Weights & Biases)
- Parallel trial execution support
