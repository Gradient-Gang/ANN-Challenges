# SubmissionGenerator Module

![Architecture Diagram](../../../../Deliverables/UML/UML_drawio.png)

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

---

### `WindowedSubmissionGenerator`
Enhanced submission generator that handles windowed model predictions. When models use WindowedModelWrapper, this generator aggregates window-level predictions to sample-level outputs for submission files.

**Configuration Parameters:**
- `model` (L.LightningModule): Trained windowed PyTorch Lightning model
- `dataloader` (torch.utils.data.DataLoader): DataLoader containing test data
- `label_mapping` (Dict[int, str] | None): Mapping from integer predictions to string labels
- `aggregation_method` (Literal): How to aggregate windowed predictions
  - `"avg_probs"`: Average softmax probabilities, then argmax (default)
  - `"majority_vote"`: Most frequent predicted class
  - `"max_confidence"`: Prediction with highest confidence
- `num_original_samples` (int | None): Number of original samples before windowing (default: None = no windowing)

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `model: L.LightningModule`<br>`dataloader: DataLoader`<br>`label_mapping: Dict[int, str]`<br>`aggregation_method: str`<br>`num_original_samples: int` | - | Initialize generator with windowed model and aggregation strategy. |
| `generate_predictions` | - | `tuple[Tensor, Tensor]` | Generate predictions and probabilities for all samples. |
| `aggregate_windowed_predictions` | `predictions: Tensor`<br>`probabilities: Tensor` | `Tensor` | Aggregate window predictions to sample-level using configured method. |
| `create_submission_file` | `output_path: str`<br>`predictions: Tensor`<br>`probabilities: Tensor` | `pd.DataFrame` | Create and save submission CSV with aggregated predictions. |
| `generate_submission` | `output_path: str` | `pd.DataFrame` | Complete workflow: predict, aggregate, and save submission. |

**Aggregation Methods:**
- **avg_probs**: Average softmax probabilities across windows, then argmax
  - Best for calibrated probability estimates
  - Smooth decision boundaries
  - Recommended for most use cases
- **majority_vote**: Most frequent class among window predictions
  - Robust to outlier windows
  - Discrete voting mechanism
  - Good for imbalanced classes
- **max_confidence**: Prediction from window with highest softmax confidence
  - Trusts most confident prediction
  - Can be sensitive to overfitting
  - Use when model calibration is strong

**Integration with FinalPipeline:**
```python
# FinalPipeline.create_submission() automatically uses WindowedSubmissionGenerator
if self.use_windowing:
    submission_df = WindowedSubmissionGenerator(
        model=best_model,
        dataloader=test_loader,
        label_mapping={0: "no_pain", 1: "low_pain", 2: "high_pain"},
        aggregation_method=self.aggregation_method,
        num_original_samples=len(test_dataset)
    ).generate_submission(output_path)
else:
    submission_df = SubmissionGenerator(
        model=best_model,
        dataloader=test_loader,
        label_mapping={0: "no_pain", 1: "low_pain", 2: "high_pain"}
    ).generate_submission(output_path)
```

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