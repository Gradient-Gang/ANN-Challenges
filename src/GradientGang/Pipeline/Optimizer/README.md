# Optimizer Module

## Description
The Optimizer module provides hyperparameter optimization capabilities using Optuna. It defines an abstract base class for optimizers and a concrete implementation for Bayesian optimization with Tree-structured Parzen Estimator (TPE) sampling.

---

## Base Class: `Optimizer`

### Description
Abstract base class defining the interface for optimization strategies.

### Methods

#### `optimize(self, architecture_builder: types.FunctionType) -> Any`
Abstract method to perform hyperparameter optimization.

**Parameters:**
- `architecture_builder` (FunctionType): Function that builds and returns a model given parameters

**Returns:**
- Optimization results (implementation-specific)

---

## Class: `OptunaOptimizer`

### Description
Concrete implementation of the Optimizer using Optuna for Bayesian hyperparameter optimization. Uses TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter search.

### Methods

#### `getParams(self, params: dict, trial: optuna.trial.BaseTrial) -> dict`
Generate hyperparameters for a single trial based on configuration.

**Parameters:**
- `params` (dict): Hyperparameter space configuration with:
  - Each key is a parameter name
  - Each value is a dict with:
    - `type` (str): Parameter type ("categ", "float", "int")
    - `params` (dict): Type-specific parameters
      - For "categ": `{"name": str, "choices": list}`
      - For "float": `{"name": str, "low": float, "high": float, "log": bool (optional)}`
      - For "int": `{"name": str, "low": int, "high": int, "step": int (optional)}`
- `trial` (optuna.trial.BaseTrial): Optuna trial object

**Returns:**
- `dict`: Dictionary of sampled hyperparameters

**Example params structure:**
```python
params = {
    "learning_rate": {
        "type": "float",
        "params": {"name": "learning_rate", "low": 1e-5, "high": 1e-2, "log": True}
    },
    "num_layers": {
        "type": "int",
        "params": {"name": "num_layers", "low": 2, "high": 5}
    },
    "activation": {
        "type": "categ",
        "params": {"name": "activation", "choices": ["ReLU", "GELU", "LeakyReLU"]}
    }
}
```

---

#### `optimize(self, architecture_builder: types.FunctionType, params: dict, n_trials: int = None) -> optuna.study.Study`
Run hyperparameter optimization.

**Parameters:**
- `architecture_builder` (FunctionType): Function that takes params dict and returns a trained model
- `params` (dict): Hyperparameter space configuration (see getParams for format)
- `n_trials` (int, optional): Number of optimization trials (None for unlimited)

**Returns:**
- `optuna.study.Study`: Completed Optuna study with results
  - Access best parameters: `study.best_params`
  - Access best value: `study.best_value`
  - Access all trials: `study.trials`

---

#### `objective(self, trial: optuna.trial.BaseTrial) -> float`
Objective function for a single optimization trial (internal method).

**Parameters:**
- `trial` (optuna.trial.BaseTrial): Current Optuna trial

**Returns:**
- `float`: Validation metric to minimize (from model's validation_step)

---

## Example Usage

### Basic Optimization

```python
from GradientGang.Pipeline.Optimizer import OptunaOptimizer
from GradientGang.Pipeline import Pipeline

# Define hyperparameter space
hyperparams = {
    "LearningRate": {
        "type": "float",
        "params": {
            "name": "LearningRate",
            "low": 1e-5,
            "high": 1e-2,
            "log": True
        }
    },
    "base_channel_size": {
        "type": "int",
        "params": {
            "name": "base_channel_size",
            "low": 32,
            "high": 128,
            "step": 32
        }
    },
    "activation_function": {
        "type": "categ",
        "params": {
            "name": "activation_function",
            "choices": ["ReLU", "GELU", "LeakyReLU"]
        }
    }
}

# Create optimizer
optimizer = OptunaOptimizer()

# Run optimization
study = optimizer.optimize(
    architecture_builder=pipeline.build_architecture,
    params=hyperparams,
    n_trials=100
)

# Get results
print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")
```

### With Pipeline Integration

```python
from GradientGang.Pipeline import Pipeline
from GradientGang.Pipeline.Optimizer import OptunaOptimizer

# Create pipeline with optimizer
pipeline = Pipeline(
    dataset=train_data,
    test=test_data,
    optimizer=OptunaOptimizer(),
    path_config="hyperparams.yaml"
)

# Run optimization (uses config from yaml)
study = pipeline.optimize()
```

## Configuration File Format (YAML)

```yaml
LearningRate:
  type: float
  params:
    name: LearningRate
    low: 0.00001
    high: 0.01
    log: true

latent_dim:
  type: int
  params:
    name: latent_dim
    low: 64
    high: 512
    step: 64

activation_function:
  type: categ
  params:
    name: activation_function
    choices:
      - ReLU
      - GELU
      - LeakyReLU
```

## Optimization Strategy

- **Sampler**: TPE (Tree-structured Parzen Estimator)
  - Bayesian optimization approach
  - Efficiently explores hyperparameter space
  - Balances exploration vs exploitation
- **Seed**: Fixed to 0 for reproducibility
- **Objective**: Minimizes validation metric (lower is better)

## Notes

- The optimizer uses a fixed random seed (0) for reproducibility
- All parameter types must have a "type" and "params" field
- The architecture_builder function must return a model with a validation_step method
- Study can be saved and resumed using Optuna's storage backends
