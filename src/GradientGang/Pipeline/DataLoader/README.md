# DataLoader Module

## Description
The DataLoader module provides a PyTorch Lightning DataModule for loading multimodal datasets combining time series and global features from CSV files. It handles label mapping, automatic train/validation/test splits, and efficient data loading with support for parallel workers and GPU memory pinning. The module is designed to work seamlessly with the Pipeline for end-to-end training workflows and supports both time series and tabular data formats.

---

## Main Classes

### `TimeSeriesAndGlobalDataset`
PyTorch Dataset class for handling combined time series and global feature data.

**Configuration Parameters:**
- `dataPath` (str): Path to CSV file containing features
- `labelsPath` (str | None): Path to CSV file containing labels (optional)
- `primaryKeyColumn` (str): Column name for sample identifier (default: "sample_index")
- `globalColumns` (list[str]): List of column names for global features
- `timeSeriesColumns` (list[str] | None): List of column names for time series features (default: auto-detect)
- `labelMapping` (list[str]): List of label names for string-to-integer mapping (default: ["no_pain", "low_pain", "high_pain"])

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `fromCSV` | `dataPath: str`<br>`labelsPath: str \| None`<br>`primaryKeyColumn: str`<br>`globalColumns: list[str]`<br>`timeSeriesColumns: list[str] \| None`<br>`labelMapping: list[str]` | `TimeSeriesAndGlobalDataset` | Create dataset from CSV files with automatic feature processing. |
| `__init__` | `time_series_data: Tensor \| None`<br>`global_data: Tensor`<br>`labels: Tensor \| None` | - | Initialize dataset with pre-loaded tensors. |
| `__len__` | - | `int` | Return number of samples in dataset. |
| `__getitem__` | `index: int` | `tuple` | Retrieve sample at specified index. |

---

### `DataModule`
PyTorch Lightning DataModule for managing dataloaders and dataset splits.

**Configuration Parameters:**
- `data_dir` (str): Directory containing data files
- `train_file_name` (str): Training data CSV filename
- `train_file_name_labels` (str): Training labels CSV filename
- `test_file_name` (str): Test data CSV filename
- `batch_size` (int): Batch size for dataloaders
- `num_workers` (int): Number of workers for parallel data loading
- `val_split` (float): Validation split ratio (default: 0.1)
- `globalFeaturesColumns` (list[str]): List of global feature column names (default: ["isPirate", "isNotPirate"])
- `primaryKeyColumn` (str): Primary key column name (default: "sample_index")
- `timeSeriesColumns` (list[str] | None): Time series column names (default: auto-detect)
- `label_mapping` (dict): Mapping from string labels to integers (default: {"no_pain": 0, "low_pain": 1, "high_pain": 2})

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict` | - | Initialize DataModule with configuration parameters. |
| `setup` | `stage: str \| None` | - | Setup datasets for specified stage ("fit", "test", or None). |
| `train_dataloader` | - | `DataLoader` | Create training dataloader with shuffling. |
| `val_dataloader` | - | `DataLoader` | Create validation dataloader without shuffling. |
| `test_dataloader` | - | `DataLoader` | Create test dataloader without shuffling. |
| `predict_dataloader` | - | `DataLoader` | Create prediction dataloader (same as test). |

---


## Data Processing Pipeline

### 1. Loading Phase
- Read CSV files using pandas
- Identify global vs. time series columns
- Extract primary key for grouping

### 2. Feature Processing
- **Global Features**: Extract first occurrence per sample (constant across time)
- **Time Series Features**: Pivot into 3D tensor (samples × features × time)
- **Auto-detection**: Columns not in globalColumns and not time/primaryKey are time series

### 3. Label Processing
- Map string labels to integers using label_mapping
- Convert to one-hot encoding (num_classes columns)
- Handle missing labels (return None for test data)

### 4. Tensor Creation
- Time series: `(num_samples, num_features, num_timesteps)` as float32
- Global features: `(num_samples, num_global_features)` as float32
- Labels: `(num_samples, num_classes)` as float32 (one-hot)

### 5. Train/Validation Split
- Random split with fixed seed (42) for reproducibility
- Configurable split ratio (default: 0.1 = 10% validation)
- Maintains sample integrity (no splitting within time series)

---

## DataLoader Features

### Automatic Optimizations
- **Shuffling**: Enabled for training, disabled for validation/test
- **Persistent Workers**: Keeps workers alive between epochs (faster)
- **Pin Memory**: Enables faster GPU transfers when CUDA available
- **Parallel Loading**: Multiple workers load data simultaneously

### Memory Management
- Time series stored as contiguous tensors for efficiency
- One-hot encoding computed once during loading
- Global features shared across architectures

### Reproducibility
- Fixed random seed (42) for train/val split
- Deterministic ordering for validation/test sets
- Consistent batch composition across runs

---

## Integration with Architectures

### Input Format
DataLoaders return batches as:
```python
(time_series_batch, global_features_batch), labels_batch
```

Where:
- `time_series_batch`: `(batch_size, num_features, num_timesteps)` or `None`
- `global_features_batch`: `(batch_size, num_global_features)`
- `labels_batch`: `(batch_size, num_classes)` one-hot encoded or `None`

### Architecture Compatibility
```python
# Direct and LightningAutoencoder expect:
def forward(self, x):
    time_series, global_features = x
    # Process both modalities
    ...
```

---

## Advanced Configuration

### Large Datasets
```python
params = {
    # ... other params
    "batch_size": 256,      # Large batches for efficiency
    "num_workers": 12,      # More parallel workers
    "val_split": 0.05,      # Smaller validation set
}
```

### Small Datasets
```python
params = {
    # ... other params
    "batch_size": 16,       # Smaller batches
    "num_workers": 0,       # Single-threaded (less overhead)
    "val_split": 0.3,       # Larger validation set
}
```

### Multi-GPU Training
```python
# DataModule automatically works with DDP
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,              # 4 GPUs
    strategy="ddp"
)
trainer.fit(model, data_module)
```

---

## Error Handling

### Missing Columns
- Global columns not in CSV are automatically filtered
- Missing time column defaults to tabular format
- Missing labels in test data handled gracefully (None returned)

### Validation
- `ParameterInterpreter` validates all required parameters
- Dimension checks ensure time series and global features match
- Label count validation ensures consistency

### Runtime Checks
```python
# RuntimeError raised if setup() not called
try:
    train_loader = data_module.train_dataloader()
except RuntimeError as e:
    print(f"Error: {e}")
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
```

---

## Performance Tips

1. **Optimal num_workers**: Start with `num_workers = 4`, increase if CPU underutilized
2. **Batch size**: Largest that fits in GPU memory (powers of 2 often best)
3. **Persistent workers**: Automatically enabled when `num_workers > 0`
4. **Pin memory**: Automatically enabled for GPU training
5. **Data preprocessing**: Use PreProcessor module before DataLoader for efficiency

---

## Notes

- Time series are automatically permuted to `(samples, features, time)` format
- Global features are extracted once per sample (not per timestep)
- Labels are one-hot encoded automatically
- Validation split uses seed=42 for reproducibility
- Test data without labels receives None labels (not dummy zeros)
- All dataloaders inherit from PyTorch Lightning's LightningDataModule
- Compatible with distributed training strategies (DDP, Horovod)
- Supports both time series and pure tabular data formats
- Column names are case-sensitive

---

## Troubleshooting

**Issue**: "Training dataset not initialized"
- **Solution**: Call `data_module.setup(stage="fit")` before accessing dataloaders

**Issue**: High CPU usage during training
- **Solution**: Reduce `num_workers` or set to 0 for single-threaded loading

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size` or use gradient accumulation

**Issue**: Labels not found
- **Solution**: Verify `label_mapping` keys match label strings in CSV

**Issue**: Inconsistent sample counts
- **Solution**: Ensure all samples in time series data have same number of timesteps

**Issue**: Global features shape mismatch
- **Solution**: Verify `globalFeaturesColumns` names match CSV column names exactly
