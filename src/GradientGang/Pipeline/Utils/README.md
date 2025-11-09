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