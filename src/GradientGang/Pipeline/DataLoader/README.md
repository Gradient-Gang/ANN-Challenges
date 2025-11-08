# DataLoader Module

## Description
The DataLoader module provides a PyTorch Lightning DataModule for loading datasets from CSV files, handling label mapping, and creating train/validation/test splits. It's designed to work seamlessly with the Pipeline for end-to-end training workflows.

## Main Class: `DataModule`

### Methods

#### `__init__(self, params: dict)`
Initialize the DataModule with configuration parameters.

**Parameters:**
- `params` (dict): Configuration dictionary with:
  - `data_dir` (str, required): Directory containing data files
  - `train_file_name` (str, required): Training data CSV filename
  - `train_file_name_labels` (str, required): Training labels CSV filename
  - `test_file_name` (str, required): Test data CSV filename
  - `batch_size` (int, required): Batch size for dataloaders
  - `num_workers` (int, required): Number of workers for data loading
  - `val_split` (float, optional): Validation split ratio (default: 0.1)
  - `label_mapping` (dict, optional): Mapping from string labels to integers
    - Default: `{"no_pain": 0, "low_pain": 1, "high_pain": 2}`

---

#### `load_from_csv(self, file_path: str, label_file_path: str = None) -> Tuple[torch.Tensor, torch.Tensor]`
Load data and labels from CSV files.

**Parameters:**
- `file_path` (str): Path to features CSV file
- `label_file_path` (str, optional): Path to labels CSV file

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: (features_tensor, labels_tensor)
  - Features are converted to float32
  - Labels are converted to long integers
  - If no labels provided, returns None for labels

**Behavior:**
- Automatically filters to numeric columns only
- Maps string labels to integers using label_mapping
- Handles missing labels gracefully

---

#### `setup(self, stage: str = None)`
Setup datasets for training, validation, and testing.

**Parameters:**
- `stage` (str, optional): Stage name ("fit", "test", or None for all)

**Behavior:**
- For "fit" stage: Loads training data and splits into train/validation
- For "test" stage: Loads test data (creates dummy labels if none provided)
- Uses fixed random seed (42) for reproducible splits

---

#### `train_dataloader(self) -> torch.utils.data.DataLoader`
Create training dataloader.

**Returns:**
- `DataLoader`: Training dataloader with shuffling enabled

**Features:**
- Shuffles data each epoch
- Uses persistent workers if num_workers > 0
- Pins memory if CUDA is available

---

#### `val_dataloader(self) -> torch.utils.data.DataLoader`
Create validation dataloader.

**Returns:**
- `DataLoader`: Validation dataloader without shuffling

---

#### `test_dataloader(self) -> torch.utils.data.DataLoader`
Create test dataloader.

**Returns:**
- `DataLoader`: Test dataloader without shuffling

---

#### `predict_dataloader(self) -> torch.utils.data.DataLoader`
Create prediction dataloader (uses test dataset).

**Returns:**
- `DataLoader`: Same as test_dataloader()

---

## Example Usage

```python
from GradientGang.Pipeline.DataLoader import DataModule

# Configuration
params = {
    "data_dir": "dataset/PirateProcessed",
    "train_file_name": "pirate_pain_train.csv",
    "train_file_name_labels": "pirate_pain_train_labels.csv",
    "test_file_name": "pirate_pain_test.csv",
    "batch_size": 32,
    "num_workers": 4,
    "val_split": 0.2,
    "label_mapping": {
        "no_pain": 0,
        "low_pain": 1,
        "high_pain": 2
    }
}

# Create DataModule
data_module = DataModule(params)

# Setup for training
data_module.setup(stage="fit")

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Use with PyTorch Lightning Trainer
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

## CSV File Format

### Training Data (features)
```csv
sample_index,feature_1,feature_2,...,feature_n
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```

### Training Labels
```csv
sample_index,label
0,no_pain
1,low_pain
2,high_pain
...
```

### Test Data
```csv
sample_index,feature_1,feature_2,...,feature_n
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```

## Notes
- Non-numeric columns (like timestamps) are automatically filtered out
- The validation split is deterministic (uses seed=42)
- For test data without labels, dummy zero labels are created
- All dataloaders are optimized for GPU training with memory pinning
