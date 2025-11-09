# Utils Module

## Description
The Utils module provides essential utility classes for parameter validation, interpretation, and configuration management across the Pipeline. It includes the `ParameterInterpreter` class for robust parameter validation with type checking, recursive nested validation, and string-to-object mapping capabilities. This module is fundamental to ensuring configuration correctness throughout the codebase, enabling flexible yet safe parameter handling for architectures, optimizers, and data loaders.

---

## Main Classes

### `ParameterInterpreter`
Utility class for validating and interpreting configuration parameters with type checking and object mapping.

**Configuration Parameters:**
- `interpretation` (dict): Mapping from string identifiers to Python objects (classes, functions, values)
- `name` (str): Name for error messages and debugging (default: "ParameterInterpreter")
- `requiredParams` (dict): Dictionary defining required parameters with expected types or nested validation rules

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `interpretation: dict`<br>`name: str`<br>`requiredParams: dict` | - | Initialize interpreter with mappings and validation rules. |
| `checkRequiredParams` | `params: dict` | `None` | Validate that all required parameters are present with correct types. |
| `interpret` | `param_name: str` | `Any` | Retrieve Python object corresponding to parameter name. |

---

## Example Usage

### Basic Parameter Validation

```python
from GradientGang.Pipeline.Utils import ParameterInterpreter

# Define validation rules
interpreter = ParameterInterpreter(
    name="ConfigValidator",
    interpretation={},
    requiredParams={
        "batch_size": int,
        "learning_rate": float,
        "num_epochs": int,
        "optimizer": str
    }
)

# Valid configuration
valid_params = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "optimizer": "adam"
}

try:
    interpreter.checkRequiredParams(valid_params)
    print("✓ Configuration valid")
except (KeyError, TypeError) as e:
    print(f"✗ Configuration error: {e}")
```

---

### String-to-Object Interpretation

```python
import torch.nn as nn
from GradientGang.Pipeline.Utils import ParameterInterpreter

# Map string names to PyTorch classes
interpreter = ParameterInterpreter(
    name="ActivationInterpreter",
    interpretation={
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU
    }
)

# Get activation class from string
activation_name = "gelu"
activation_class = interpreter.interpret(activation_name)

# Instantiate the activation
activation = activation_class()
print(f"Created activation: {activation}")
# Output: Created activation: GELU()
```

---

### Nested Parameter Validation

```python
# Validate nested configuration dictionaries
interpreter = ParameterInterpreter(
    name="ModelConfigValidator",
    interpretation={},
    requiredParams={
        "model_name": str,
        "encoder_config": {
            "input_dim": int,
            "hidden_dim": int,
            "num_layers": int
        },
        "decoder_config": {
            "latent_dim": int,
            "output_dim": int,
            "activation": str
        },
        "training_config": {
            "learning_rate": float,
            "batch_size": int,
            "max_epochs": int
        }
    }
)

# Configuration to validate
config = {
    "model_name": "autoencoder",
    "encoder_config": {
        "input_dim": 128,
        "hidden_dim": 64,
        "num_layers": 3
    },
    "decoder_config": {
        "latent_dim": 32,
        "output_dim": 128,
        "activation": "relu"
    },
    "training_config": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 100
    }
}

# Validate entire nested structure
interpreter.checkRequiredParams(config)
print("✓ Nested configuration validated successfully")
```

---

### Combined Validation and Interpretation

```python
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW

# Setup interpreter with both validation and interpretation
interpreter = ParameterInterpreter(
    name="OptimizerBuilder",
    interpretation={
        "adam": Adam,
        "sgd": SGD,
        "adamw": AdamW,
        "relu": nn.ReLU,
        "gelu": nn.GELU
    },
    requiredParams={
        "optimizer_type": str,
        "learning_rate": float,
        "activation": str,
        "model_config": dict
    }
)

# Configuration
params = {
    "optimizer_type": "adam",
    "learning_rate": 0.001,
    "activation": "gelu",
    "model_config": {"hidden_dim": 256}
}

# Validate
interpreter.checkRequiredParams(params)

# Interpret and use
optimizer_class = interpreter.interpret(params["optimizer_type"])
activation_class = interpreter.interpret(params["activation"])

print(f"Optimizer: {optimizer_class}")
print(f"Activation: {activation_class}")
```

---



## Validation Strategies

### 1. Strict Validation
```python
# All parameters required, no defaults
interpreter = ParameterInterpreter(
    name="StrictValidator",
    interpretation={},
    requiredParams={
        "param1": int,
        "param2": float,
        "param3": str
    }
)
```

### 2. Nested Validation
```python
# Validate hierarchical configurations
interpreter = ParameterInterpreter(
    name="NestedValidator",
    interpretation={},
    requiredParams={
        "level1": {
            "level2": {
                "level3": int
            }
        }
    }
)
```

### 3. Partial Validation
```python
# Validate only critical parameters
interpreter = ParameterInterpreter(
    name="PartialValidator",
    interpretation={},
    requiredParams={
        "critical_param": int
        # Other params optional
    }
)
```

---

## Error Messages

### KeyError Examples
```python
# Missing parameter
# Output: ConfigValidator: Required parameter 'learning_rate' not found in provided parameters.
```

### TypeError Examples
```python
# Wrong type
# Output: ConfigValidator: Parameter 'batch_size' must be of type int.
```

### Interpretation Error
```python
# Unknown parameter name
# Output: ActivationInterpreter: 'unknown_activation' not found in interpretation dictionary.
```

---

## Integration with Other Modules

### With OptunaOptimizer
```python
from GradientGang.Pipeline.Optimizer import OptunaOptimizer
from GradientGang.Pipeline.Utils import ParameterInterpreter

# OptunaOptimizer uses ParameterInterpreter internally
# for validating parameter space configuration
```

### With DataLoader
```python
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Utils import ParameterInterpreter

# Validate DataModule parameters before creation
dataloader_validator = ParameterInterpreter(
    name="DataModuleValidator",
    interpretation={},
    requiredParams={
        "data_dir": str,
        "batch_size": int,
        "num_workers": int
    }
)

dataloader_validator.checkRequiredParams(data_params)
data_module = DataModule(data_params)
```

### With Architectures
```python
from GradientGang.Pipeline.Architectures import Direct
from GradientGang.Pipeline.Utils import ParameterInterpreter

# Validate architecture parameters
arch_validator = ParameterInterpreter(
    name="ArchitectureValidator",
    interpretation={},
    requiredParams={
        "input_dim": int,
        "hidden_dim": int,
        "num_layers": int
    }
)

arch_validator.checkRequiredParams(arch_params)
model = Direct(**arch_params)
```

---

## Performance Considerations

### Validation Overhead
- Validation adds minimal overhead (microseconds)
- Perform at initialization, not in training loops
- Cache interpreter instances for reuse

### Best Practices
```python
# Good: Validate once at initialization
class MyModel:
    def __init__(self, config):
        self.validator = ParameterInterpreter(...)
        self.validator.checkRequiredParams(config)
        # ... rest of initialization

# Bad: Validate in forward pass
class MyModel:
    def forward(self, x):
        self.validator.checkRequiredParams(...)  # Too slow!
        # ... forward logic
```

---

## Troubleshooting

**Issue**: "Required parameter 'X' not found"
- **Solution**: Add missing parameter to configuration dictionary

**Issue**: "Parameter 'X' must be of type Y"
- **Solution**: Convert parameter to correct type before validation

**Issue**: Nested validation not working
- **Solution**: Ensure nested structure in requiredParams matches config structure

**Issue**: "Parameter name not found in interpretation dictionary"
- **Solution**: Add parameter name to interpretation dictionary or check spelling

**Issue**: Validation passes but program crashes later
- **Solution**: Add more specific validation rules or custom validation logic

**Issue**: Too many validation rules to maintain
- **Solution**: Use configuration files (YAML/JSON) to store validation rules

---

## Notes

- Type checking uses `isinstance()` for runtime validation
- Supports recursive validation for arbitrarily nested dictionaries
- Error messages include interpreter name for easy debugging
- Interpretation dictionary can map to any Python object (classes, functions, values)
- Required params can be types (for type checking) or dicts (for nested validation)
- Validation happens at runtime, not at import time
- Thread-safe for concurrent validation
- No external dependencies beyond Python standard library
- Used extensively throughout the Pipeline codebase
- Supports all Python built-in types (int, float, str, list, dict, bool, etc.)

---

## Best Practices

1. **Name Your Interpreters**: Use descriptive names for clear error messages
2. **Validate Early**: Check parameters at initialization, not during execution
3. **Reuse Interpreters**: Create once, use multiple times for efficiency
4. **Comprehensive Rules**: Define all required parameters explicitly
5. **Nested Validation**: Use nested dicts for complex configurations
6. **Clear Mappings**: Use intuitive string keys in interpretation dictionaries
7. **Error Handling**: Always wrap validation in try-except for graceful failures
8. **Documentation**: Document required parameters in docstrings
9. **Type Hints**: Use type hints alongside ParameterInterpreter for better IDE support
10. **Test Validation**: Write unit tests for parameter validation logic

---

## Future Enhancements

- Optional parameter validation with default values
- Value range validation (min/max constraints)
- Custom validation functions
- Automatic documentation generation from requiredParams
- JSON schema export/import
- Validation rule inheritance
- Soft validation (warnings instead of errors)
- Parameter dependency validation (if A then B required)
- Automatic type coercion with warnings
- Integration with dataclasses and Pydantic models
- Validation performance profiling
- Configuration file format support (YAML, JSON, TOML)
