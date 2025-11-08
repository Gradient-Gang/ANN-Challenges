# Utils Module

This module provides utility classes for parameter validation and submission file generation.

---

## ParameterInterpreter

### Description
A utility class for validating and interpreting configuration parameters. Provides type checking, required parameter validation, and mapping of string identifiers to Python objects/types.

### Methods

#### `__init__(self, interpretation: dict, name: str = "ParameterInterpreter", requiredParams: dict = {})`
Initialize the parameter interpreter.

**Parameters:**
- `interpretation` (dict): Mapping from string identifiers to Python objects (classes, functions, values)
  - Example: `{"ReLU": nn.ReLU, "GELU": nn.GELU}`
- `name` (str, optional): Name for error messages (default: "ParameterInterpreter")
- `requiredParams` (dict, optional): Dictionary defining required parameters with:
  - Key: parameter name
  - Value: expected type (type object) or nested dict for recursive validation

---

#### `checkRequiredParams(self, params: dict)`
Validate that all required parameters are present with correct types.

**Parameters:**
- `params` (dict): Parameters to validate

**Raises:**
- `KeyError`: If a required parameter is missing
- `TypeError`: If a parameter has incorrect type

**Example:**
```python
interpreter = ParameterInterpreter(
    interpretation={},
    requiredParams={
        "batch_size": int,
        "learning_rate": float,
        "model_config": dict
    }
)
interpreter.checkRequiredParams(params)  # Validates params
```

---

#### `interpret(self, param_name: str) -> Any`
Retrieve the Python object corresponding to a parameter name.

**Parameters:**
- `param_name` (str): Name to look up in interpretation dictionary

**Returns:**
- `Any`: The corresponding Python object (class, function, value)

**Raises:**
- `KeyError`: If param_name not found in interpretation dictionary

**Example:**
```python
interpreter = ParameterInterpreter(
    interpretation={"ReLU": nn.ReLU, "GELU": nn.GELU}
)
activation = interpreter.interpret("ReLU")  # Returns nn.ReLU
```

---

## SubmissionGenerator

### Description
Generates submission CSV files for classification challenges. Takes a trained model and test dataloader, performs predictions, and creates properly formatted submission files with sample indices and predicted labels.

### Methods

#### `__init__(self, model: L.LightningModule, dataloader: torch.utils.data.DataLoader, label_mapping: Dict[int, str] = None)`
Initialize the submission generator.

**Parameters:**
- `model` (L.LightningModule): Trained PyTorch Lightning model
- `dataloader` (DataLoader): DataLoader containing test data
- `label_mapping` (Dict[int, str], optional): Mapping from integer labels to string labels
  - Default: `{0: "no_pain", 1: "low_pain", 2: "high_pain"}`

---

#### `generate_predictions(self) -> torch.Tensor`
Generate predictions for all samples in the dataloader.

**Returns:**
- `torch.Tensor`: Tensor of predicted class indices (integers)

**Behavior:**
- Automatically moves data to model's device
- Handles different model output formats (single tensor or tuple)
- Uses torch.no_grad() for efficiency

---

#### `create_submission_file(self, output_path: str, predictions: torch.Tensor = None) -> pd.DataFrame`
Create a submission CSV file with predictions.

**Parameters:**
- `output_path` (str): Path where the submission CSV will be saved
- `predictions` (torch.Tensor, optional): Pre-computed predictions (generates if None)

**Returns:**
- `pd.DataFrame`: DataFrame containing the submission data

**Output Format:**
```csv
sample_index,label
000,no_pain
001,low_pain
002,high_pain
...
```

**Prints:**
- Success message with file path
- Total number of samples
- Label distribution (value counts)

---

#### `generate_submission(self, output_path: str = "submission.csv") -> pd.DataFrame`
Complete workflow: generate predictions and create submission file.

**Parameters:**
- `output_path` (str, optional): Path where submission CSV will be saved (default: "submission.csv")

**Returns:**
- `pd.DataFrame`: DataFrame containing the submission data

---

### Convenience Function: `generate_submission`

```python
def generate_submission(
    model: L.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    output_path: str = "submission.csv",
    label_mapping: Dict[int, str] = None
) -> pd.DataFrame
```

Quick one-call function to generate submission files.

**Parameters:**
- Same as SubmissionGenerator.__init__ plus output_path

**Returns:**
- `pd.DataFrame`: Submission DataFrame

---

## Example Usage

### ParameterInterpreter

```python
from GradientGang.Pipeline.Utils import ParameterInterpreter
import torch.nn as nn

# Define interpretation mapping
interpreter = ParameterInterpreter(
    name="ArchitectureInterpreter",
    interpretation={
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "Conv2d": nn.Conv2d,
        "Linear": nn.Linear
    },
    requiredParams={
        "activation": str,
        "num_layers": int,
        "layer_config": dict
    }
)

# Validate parameters
params = {
    "activation": "GELU",
    "num_layers": 3,
    "layer_config": {...}
}
interpreter.checkRequiredParams(params)

# Get activation class
activation_cls = interpreter.interpret("GELU")  # Returns nn.GELU
activation = activation_cls()  # Instantiate
```

### SubmissionGenerator (Class-based)

```python
from GradientGang.Pipeline.Utils import SubmissionGenerator

# Create generator
generator = SubmissionGenerator(
    model=trained_model,
    dataloader=test_dataloader,
    label_mapping={0: "no_pain", 1: "low_pain", 2: "high_pain"}
)

# Generate submission file
submission_df = generator.generate_submission("my_submission.csv")
# Output:
# Submission file created successfully at: my_submission.csv
# Total samples: 150
# 
# Label distribution:
# no_pain      80
# low_pain     45
# high_pain    25
```

### SubmissionGenerator (Function-based)

```python
from GradientGang.Pipeline.Utils.SubmissionGenerator import generate_submission

# One-line submission generation
submission_df = generate_submission(
    model=trained_model,
    dataloader=test_dataloader,
    output_path="submission.csv"
)
```

### Custom Label Mapping

```python
# For different classification tasks
custom_mapping = {
    0: "class_a",
    1: "class_b",
    2: "class_c",
    3: "class_d"
}

generator = SubmissionGenerator(
    model=model,
    dataloader=test_loader,
    label_mapping=custom_mapping
)
submission = generator.generate_submission("custom_submission.csv")
```

## Notes

### ParameterInterpreter
- Supports recursive validation for nested dictionaries
- Error messages include the interpreter name for debugging
- Commonly used throughout the codebase for config validation

### SubmissionGenerator
- Automatically handles GPU/CPU device management
- Works with models that return tuples (e.g., autoencoders returning predictions + reconstructions)
- Sample indices are zero-padded (000, 001, etc.) for consistency
- Prints label distribution to help identify class imbalances
