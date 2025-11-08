# DataLoader Module

## Overview

The DataLoader module provides a PyTorch Lightning-compatible data loading system for the GradientGang pipeline. It's designed to load CSV-based datasets with separate feature and label files, with automatic train/validation splitting.

## Features

- ✅ **PyTorch Lightning Integration**: Fully compatible with `LightningDataModule`
- ✅ **CSV Support**: Load data from CSV files with automatic tensor conversion
- ✅ **Automatic Train/Val Split**: Configurable validation split ratio
- ✅ **Label Mapping**: Convert string labels to integer indices
- ✅ **Parameter Validation**: Built-in parameter checking with `ParameterInterpreter`
- ✅ **Multi-worker Support**: Efficient data loading with configurable workers
- ✅ **GPU Optimization**: Automatic pin_memory for CUDA devices

## Class: DataModule

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data_dir` | str | Yes | - | Directory containing data files |
| `train_file_name` | str | Yes | - | Training features CSV filename |
| `train_file_name_labels` | str | Yes | - | Training labels CSV filename |
| `test_file_name` | str | Yes | - | Test features CSV filename |
| `batch_size` | int | Yes | - | Batch size for dataloaders |
| `num_workers` | int | Yes | - | Number of worker processes |
| `val_split` | float | No | 0.1 | Fraction of training data for validation |
| `label_mapping` | dict | No | See below | Mapping from string labels to integers |

### Default Label Mapping

```python
{
    "no_pain": 0,
    "low_pain": 1,
    "high_pain": 2
}
```

