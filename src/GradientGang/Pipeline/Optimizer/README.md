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
