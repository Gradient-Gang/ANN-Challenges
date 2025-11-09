# SubmissionGenerator Module

## Description
The SubmissionGenerator module provides automated submission file generation for classification challenges. It takes trained PyTorch Lightning models and test dataloaders, performs batch predictions, and creates properly formatted CSV files with sample indices and predicted labels. The module handles various model output formats, supports custom label mappings, and includes convenience functions for quick one-line submission generation. It's designed to seamlessly integrate with the Pipeline for end-to-end workflows from training to submission.

---

## Main Classes

### `SubmissionGenerator`
Main class for generating submission files from model predictions.

**Configuration Parameters:**
- `model` (L.LightningModule): Trained PyTorch Lightning model
- `dataloader` (torch.utils.data.DataLoader): DataLoader containing test data
- `label_mapping` (Dict[int, str] | None): Mapping from integer predictions to string labels (default: {0: "no_pain", 1: "low_pain", 2: "high_pain"})

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `model: L.LightningModule`<br>`dataloader: DataLoader`<br>`label_mapping: Dict[int, str] \| None` | - | Initialize generator with model and data. |
| `generate_predictions` | - | `torch.Tensor` | Generate predictions for all samples in dataloader. |
| `create_submission_file` | `output_path: str`<br>`predictions: Tensor \| None` | `pd.DataFrame` | Create and save submission CSV file. |
| `generate_submission` | `output_path: str` | `pd.DataFrame` | Complete workflow: predict and save submission. |

---

## Convenience Functions

### `generate_submission`
One-line function to generate submission files without explicit class instantiation.

**Parameters:**
- `model` (L.LightningModule): Trained PyTorch Lightning model
- `dataloader` (torch.utils.data.DataLoader): Test data loader
- `output_path` (str): CSV output path (default: "submission.csv")
- `label_mapping` (Dict[int, str] | None): Custom label mapping (optional)

**Returns:**
- `pd.DataFrame`: Submission dataframe with sample indices and labels

---

## Example Usage

### Basic Submission Generation

```python
from GradientGang.Pipeline.SubmissionGenerator import SubmissionGenerator
import pytorch_lightning as L

# Assuming you have a trained model and test dataloader
trained_model = L.LightningModule.load_from_checkpoint("best_model.ckpt")
test_loader = data_module.test_dataloader()

# Create generator
generator = SubmissionGenerator(
    model=trained_model,
    dataloader=test_loader
)

# Generate submission file
submission_df = generator.generate_submission("submission.csv")

print(f"Generated submission with {len(submission_df)} predictions")
print(submission_df.head())
```

---

### Using Convenience Function

```python
from GradientGang.Pipeline.SubmissionGenerator import generate_submission

# One-line submission generation
submission_df = generate_submission(
    model=trained_model,
    dataloader=test_loader,
    output_path="my_submission.csv"
)

# View results
print(submission_df)
```

---

### Custom Label Mapping

```python
# Define custom label mapping for different classification tasks
custom_mapping = {
    0: "class_a",
    1: "class_b",
    2: "class_c",
    3: "class_d"
}

generator = SubmissionGenerator(
    model=trained_model,
    dataloader=test_loader,
    label_mapping=custom_mapping
)

submission_df = generator.generate_submission("custom_submission.csv")
```

---

### Complete Pipeline Integration

```python
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.Architectures import Direct
from GradientGang.Pipeline.SubmissionGenerator import generate_submission
import pytorch_lightning as L

# 1. Setup data
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

# 2. Train model
model = Direct(data_module=data_module, hidden_dim=256, num_layers=3)
trainer = L.Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model, data_module)

# 3. Setup test data
data_module.setup(stage="test")
test_loader = data_module.test_dataloader()

# 4. Generate submission
submission_df = generate_submission(
    model=model,
    dataloader=test_loader,
    output_path="final_submission.csv"
)

print(f"Submission saved with {len(submission_df)} predictions")
```

---

### Multiple Submissions from Different Models

```python
# Compare multiple models
models = {
    "direct": Direct.load_from_checkpoint("direct_best.ckpt"),
    "autoencoder": LightningAutoencoder.load_from_checkpoint("ae_best.ckpt"),
    "ensemble": EnsembleModel.load_from_checkpoint("ensemble_best.ckpt")
}

for model_name, model in models.items():
    submission_df = generate_submission(
        model=model,
        dataloader=test_loader,
        output_path=f"submission_{model_name}.csv"
    )
    print(f"{model_name}: Generated {len(submission_df)} predictions")
```

---


## Output File Format

### Submission CSV Structure

```csv
sample_index,label
000,no_pain
001,low_pain
002,high_pain
003,no_pain
004,low_pain
```

**Format Specifications:**
- **sample_index**: Zero-padded 3-digit integers (000, 001, 002, ...)
- **label**: String labels from label_mapping
- **Encoding**: UTF-8
- **Line endings**: System default
- **Headers**: Included (required)

---

### Sample Index Formatting

```python
# Automatic zero-padding for sample indices
num_samples = 5
sample_indices = [f"{i:03d}" for i in range(num_samples)]
# Result: ['000', '001', '002', '003', '004']

# For larger datasets, padding adjusts automatically
num_samples = 1000
sample_indices = [f"{i:03d}" for i in range(num_samples)]
# Result: ['000', '001', ..., '999']
```

**Note**: Current implementation uses 3-digit padding. For datasets with >999 samples, consider modifying the format string.

---

## Prediction Pipeline

### 1. Model Preparation
- Set model to evaluation mode (`model.eval()`)
- Disable gradient computation (`torch.no_grad()`)
- Verify model is on correct device

### 2. Batch Processing
- Iterate through dataloader batches
- Extract features from batch (handle with/without labels)
- Move features to model device
- Generate predictions

### 3. Output Handling
- Support for direct logits output
- Support for tuple output (predictions, auxiliary)
- Extract logits from various output formats
- Apply argmax to get class indices

### 4. Result Aggregation
- Collect predictions from all batches
- Concatenate into single tensor
- Move to CPU for DataFrame creation
- Convert to numpy for pandas compatibility

### 5. File Creation
- Map integer predictions to string labels
- Create zero-padded sample indices
- Construct pandas DataFrame
- Save to CSV without row indices

---

## Device Management

### Automatic Device Detection
```python
# Generator automatically detects model device
device = next(model.parameters()).device

# Moves input features to model device
features = features.to(device)

# Moves predictions back to CPU for DataFrame
predictions = predicted_classes.cpu()
```

### Handling Models Without Parameters
```python
# For models with no learnable parameters
try:
    device = next(model.parameters()).device
except StopIteration:
    # Model has no parameters, use CPU
    pass
```

### Multi-GPU Inference
```python
# For DataParallel models
if isinstance(model, torch.nn.DataParallel):
    model = model.module  # Extract underlying model

generator = SubmissionGenerator(model, test_loader)
submission_df = generator.generate_submission("multi_gpu_submission.csv")
```

---

## Label Mapping

### Default Mapping
```python
default_mapping = {
    0: "no_pain",
    1: "low_pain",
    2: "high_pain"
}
```

### Custom Mappings

**Binary Classification:**
```python
binary_mapping = {
    0: "negative",
    1: "positive"
}
```

**Multi-class Classification:**
```python
multiclass_mapping = {
    0: "cat",
    1: "dog",
    2: "bird",
    3: "fish",
    4: "horse"
}
```

**Named Classes:**
```python
named_mapping = {
    0: "benign",
    1: "malignant",
    2: "uncertain"
}
```

### Mapping Validation
```python
# Ensure all predicted classes have mappings
predictions = generator.generate_predictions()
unique_predictions = torch.unique(predictions).tolist()

for pred in unique_predictions:
    if pred not in label_mapping:
        print(f"Warning: No mapping for class {pred}")
```

---

## Error Handling

### Missing Labels
```python
# Handle predictions not in mapping
try:
    predicted_labels = [label_mapping[pred] for pred in predictions_np]
except KeyError as e:
    print(f"Error: Prediction {e} not in label_mapping")
    print(f"Available mappings: {label_mapping}")
    raise
```

### Empty Dataloader
```python
# Generator handles empty dataloaders gracefully
if len(dataloader) == 0:
    predictions = torch.tensor([], dtype=torch.long)
    submission_df = pd.DataFrame({'sample_index': [], 'label': []})
    submission_df.to_csv(output_path, index=False)
```

### Model Output Validation
```python
# Validate model outputs
outputs = model(features)

if isinstance(outputs, tuple):
    logits = outputs[0]
elif isinstance(outputs, dict):
    logits = outputs.get('logits', outputs.get('predictions'))
else:
    logits = outputs

# Check dimensions
assert logits.dim() == 2, f"Expected 2D logits, got {logits.dim()}D"
assert logits.size(1) == num_classes, f"Expected {num_classes} classes, got {logits.size(1)}"
```

---

## Integration Patterns

### With DataModule

```python
from GradientGang.Pipeline.DataLoader import DataModule
from GradientGang.Pipeline.SubmissionGenerator import generate_submission

# Setup data
data_module = DataModule(params)
data_module.setup(stage="test")

# Generate submission
submission_df = generate_submission(
    model=trained_model,
    dataloader=data_module.test_dataloader(),
    output_path="submission.csv"
)
```

---

### With Trainer Predictions

```python
# Use Lightning Trainer's predict method
predictions_list = trainer.predict(model, data_module)

# Extract predictions from trainer output
# (depends on model's predict_step implementation)
all_predictions = torch.cat([pred for pred in predictions_list], dim=0)

# Create submission
generator = SubmissionGenerator(model, test_loader)
submission_df = generator.create_submission_file(
    "trainer_submission.csv",
    predictions=all_predictions
)
```

---

### With Pipeline

```python
from GradientGang.Pipeline import Pipeline

# Complete pipeline with submission
pipeline = Pipeline(config)
trained_model = pipeline.train()

# Generate submission as final step
submission_df = generate_submission(
    model=trained_model,
    dataloader=pipeline.test_loader,
    output_path="pipeline_submission.csv"
)
```

---

## Advanced Features

### Prediction Confidence Scores

```python
# Extract confidence scores along with predictions
model.eval()
all_predictions = []
all_confidences = []

with torch.no_grad():
    for batch in test_loader:
        features = batch[0] if isinstance(batch, (list, tuple)) else batch
        device = next(model.parameters()).device
        features = features.to(device)
        
        outputs = model(features)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get predictions and confidence
        confidence, predicted_classes = torch.max(probs, dim=1)
        
        all_predictions.append(predicted_classes.cpu())
        all_confidences.append(confidence.cpu())

predictions = torch.cat(all_predictions, dim=0)
confidences = torch.cat(all_confidences, dim=0)

# Create submission with confidence column
generator = SubmissionGenerator(model, test_loader)
submission_df = generator.create_submission_file("submission.csv", predictions)

# Add confidence column
submission_df['confidence'] = confidences.numpy()
submission_df.to_csv("submission_with_confidence.csv", index=False)
```

---

### Thresholded Predictions

```python
# Only include high-confidence predictions
confidence_threshold = 0.8

model.eval()
all_predictions = []
all_confidences = []

with torch.no_grad():
    for batch in test_loader:
        features = batch[0] if isinstance(batch, (list, tuple)) else batch
        device = next(model.parameters()).device
        features = features.to(device)
        
        outputs = model(features)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)
        
        confidence, predicted_classes = torch.max(probs, dim=1)
        
        # Apply threshold
        low_confidence_mask = confidence < confidence_threshold
        predicted_classes[low_confidence_mask] = -1  # Mark as uncertain
        
        all_predictions.append(predicted_classes.cpu())
        all_confidences.append(confidence.cpu())

predictions = torch.cat(all_predictions, dim=0)
confidences = torch.cat(all_confidences, dim=0)

# Create custom mapping with uncertainty
label_mapping_with_uncertain = {
    -1: "uncertain",
    0: "no_pain",
    1: "low_pain",
    2: "high_pain"
}

generator = SubmissionGenerator(model, test_loader, label_mapping_with_uncertain)
submission_df = generator.create_submission_file("thresholded_submission.csv", predictions)
```

---

### Test-Time Augmentation (TTA)

```python
# Apply augmentations at test time for robust predictions
import torch.nn.functional as F

num_augmentations = 5
all_augmented_probs = []

model.eval()
with torch.no_grad():
    for aug_idx in range(num_augmentations):
        aug_probs = []
        
        for batch in test_loader:
            features = batch[0] if isinstance(batch, (list, tuple)) else batch
            device = next(model.parameters()).device
            features = features.to(device)
            
            # Apply augmentation (example: add noise)
            if aug_idx > 0:
                noise = torch.randn_like(features) * 0.01
                features = features + noise
            
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = F.softmax(logits, dim=1)
            aug_probs.append(probs.cpu())
        
        all_augmented_probs.append(torch.cat(aug_probs, dim=0))

# Average probabilities across augmentations
avg_probs = torch.stack(all_augmented_probs, dim=0).mean(dim=0)
tta_predictions = torch.argmax(avg_probs, dim=1)

# Create submission
generator = SubmissionGenerator(model, test_loader)
submission_df = generator.create_submission_file(
    "tta_submission.csv",
    predictions=tta_predictions
)
```

---

## Performance Optimization

### Batch Size Tuning
```python
# Larger batch sizes for faster inference
test_params = {
    **data_params,
    "batch_size": 128,  # Increase for inference
    "num_workers": 8     # More workers
}

test_data_module = DataModule(test_params)
test_data_module.setup(stage="test")
test_loader = test_data_module.test_dataloader()

# Faster submission generation
submission_df = generate_submission(model, test_loader, "fast_submission.csv")
```

### Half Precision Inference
```python
# Use FP16 for faster inference on GPU
model = model.half()  # Convert to FP16

# Ensure inputs are also FP16
# Note: May require modifications to SubmissionGenerator
```

### Compiled Models
```python
# Use torch.compile for faster inference (PyTorch 2.0+)
compiled_model = torch.compile(model)

submission_df = generate_submission(
    model=compiled_model,
    dataloader=test_loader,
    output_path="compiled_submission.csv"
)
```

---

## Validation and Testing

### Validate Submission Format
```python
def validate_submission(submission_df, expected_samples):
    """Validate submission DataFrame format."""
    # Check columns
    assert list(submission_df.columns) == ['sample_index', 'label'], \
        "Invalid column names"
    
    # Check number of samples
    assert len(submission_df) == expected_samples, \
        f"Expected {expected_samples} samples, got {len(submission_df)}"
    
    # Check sample indices
    expected_indices = [f"{i:03d}" for i in range(expected_samples)]
    assert list(submission_df['sample_index']) == expected_indices, \
        "Invalid sample indices"
    
    # Check no missing values
    assert not submission_df.isnull().any().any(), \
        "Missing values detected"
    
    print("✓ Submission format validated successfully")

# Use validation
submission_df = generate_submission(model, test_loader, "submission.csv")
validate_submission(submission_df, expected_samples=len(test_loader.dataset))
```

---

### Cross-Validation Submissions

```python
# Generate submissions from multiple folds
fold_submissions = []

for fold_idx, model_path in enumerate(fold_model_paths):
    # Load fold model
    model = MyModel.load_from_checkpoint(model_path)
    
    # Generate predictions
    generator = SubmissionGenerator(model, test_loader)
    predictions = generator.generate_predictions()
    fold_submissions.append(predictions)

# Average predictions across folds
stacked_predictions = torch.stack(fold_submissions, dim=0)
avg_predictions, _ = torch.mode(stacked_predictions, dim=0)

# Create final submission
generator = SubmissionGenerator(model, test_loader)
submission_df = generator.create_submission_file(
    "cv_ensemble_submission.csv",
    predictions=avg_predictions
)
```

---

## Troubleshooting

**Issue**: "IndexError: index out of range in self"
- **Solution**: Check that label_mapping includes all predicted class indices

**Issue**: Predictions are all the same class
- **Solution**: Verify model is trained properly, check for data leakage, validate test data preprocessing

**Issue**: Model outputs wrong shape
- **Solution**: Ensure model's forward method returns logits of shape (batch_size, num_classes)

**Issue**: Device mismatch errors
- **Solution**: Generator handles device placement automatically, ensure model has parameters or handle parameter-less models

**Issue**: Memory errors during prediction
- **Solution**: Reduce batch size in test dataloader, use gradient checkpointing

**Issue**: Submission file has wrong number of samples
- **Solution**: Verify test dataloader size matches expected submission size

**Issue**: Sample indices not zero-padded correctly
- **Solution**: Current implementation uses 3-digit padding, modify format string for different requirements

**Issue**: Labels are integers instead of strings
- **Solution**: Ensure label_mapping is provided and maps integers to strings

---

## Notes

- Model is automatically set to evaluation mode (`model.eval()`)
- Gradient computation is disabled for efficiency (`torch.no_grad()`)
- Device placement is handled automatically
- Predictions are moved to CPU before DataFrame creation
- Default label mapping: {0: "no_pain", 1: "low_pain", 2: "high_pain"}
- Sample indices are zero-padded to 3 digits (000, 001, 002, ...)
- Handles various model output formats (direct logits, tuples, dicts)
- Empty dataloaders result in empty submission files
- CSV files saved without row indices (`index=False`)
- Compatible with all PyTorch Lightning models
- Supports batch processing for memory efficiency
- Thread-safe for parallel dataloader workers
- No data augmentation applied by default (test set)

---

## Best Practices

1. **Validate Model First**: Check validation performance before generating submissions
2. **Verify Format**: Use validation functions to check submission format
3. **Save Predictions**: Store raw predictions for ensemble creation
4. **Multiple Seeds**: Generate submissions from multiple training runs
5. **Confidence Thresholds**: Consider filtering low-confidence predictions
6. **Ensemble Methods**: Combine predictions from multiple models
7. **Test-Time Augmentation**: Use TTA for more robust predictions
8. **Version Control**: Include model checkpoint path in submission filename
9. **Reproducibility**: Set seeds before prediction for deterministic results
10. **Documentation**: Log model configuration and performance with submission

---

## Future Enhancements

- Support for different sample index formats (configurable padding)
- Built-in confidence score output
- Automatic ensemble submission generation
- Test-time augmentation integration
- Prediction uncertainty quantification
- Support for regression tasks
- Multi-label classification support
- Submission metadata tracking
- Automatic format validation
- Integration with competition platforms (Kaggle, CodaLab)
- Parallel batch processing for faster inference
- Automatic model selection based on validation performance
- Submission versioning and comparison tools
