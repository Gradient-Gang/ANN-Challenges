# PreProcessing Module

![Architecture Diagram](../../../Deliverables/UML/UML_drawio.png)

## Description
The PreProcessing module provides a comprehensive data preprocessing pipeline for the pirate pain multivariate time series dataset. It handles data loading, feature engineering (categorical mapping, global feature extraction), Z-score normalization, dimensionality reduction (PCA/POD, supervised feature selection), class weight computation, and visualization. The module transforms raw time series data into processed time series and global features suitable for neural network training. All operations are configured via YAML files for reproducible preprocessing workflows.

## Main Class: `PreProcessor`

### Configuration Parameters

**Path Configuration:**
- `path_raw_data` (str): Directory containing raw CSV files
- `path_processed_data` (str): Directory for saving processed files
- `name_train_file` (str): Training data filename (default: "train.csv")
- `name_test_file` (str): Test data filename (default: "test.csv")
- `name_train_labels_file` (str): Training labels filename (default: "train_labels.csv")

**Feature Engineering:**
- `drop_all_is_pirate` (bool): Drop all pirate features (n_legs, n_hands, n_eyes) (default: False)
- `one_hot_encode` (bool): One-hot encode isPirate feature (default: False)
- `extract_global_features` (bool): Extract statistical/trend global features from time series (default: False)
- `global_features_to_extract` (list[str]): Types of global features ["statistical", "trend", "domain"]
- `columns_excluded_from_normalization` (list[str]): Columns to exclude from normalization

**Dimensionality Reduction:**
- `PCA` (bool): Apply PCA/POD to time series temporal dimension (default: False)
- `explained_variance` (float): Cumulative explained variance ratio for PCA (default: 0.95)
- `feature_selection` (bool): Apply supervised feature selection to global features (default: False)
- `feature_selection_params` (dict): Parameters for FeatureSelector (variance threshold, correlation, top-k RF/MI, etc.)

**Visualization:**
- `verbose` (bool): Display plots and progress information (default: True)

---

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `fromYAML` | `path: str` | `PreProcessor` | Static method to build PreProcessor from YAML configuration file. |
| `__init__` | `params: dict` | - | Initialize PreProcessor with configuration dictionary. |
| `load_data` | `file_name: str` | `pd.DataFrame` | Load CSV file from `path_raw_data` directory. |
| `save_data` | `data: pd.DataFrame\|np.ndarray`<br>`file_name: str` | - | Save DataFrame or NumPy array as CSV to `path_processed_data`. |
| `remove_last_column` | `data: pd.DataFrame` | `pd.DataFrame` | Remove trailing column from DataFrame. |
| `handle_is_pirate_features` | `data: pd.DataFrame` | `pd.DataFrame` | Map n_eyes to isPirate binary indicator, drop n_legs/n_hands, optional one-hot encoding. |
| `normalize_per_process` | `training_data: pd.DataFrame`<br>`test_data: pd.DataFrame` | `tuple[pd.DataFrame, pd.DataFrame]` | Z-score normalization using training statistics (mean=0, std=1) per feature. |
| `extract_global_features` | `data: pd.DataFrame`<br>`feature_types: list[str]` | `pd.DataFrame` | Extract statistical (mean, std, min, max, range), trend (slope, first, last, change), and domain-specific global features from time series. |
| `plot_one_time_series` | `data: pd.DataFrame`<br>`number: int` | - | Visualize multiple time series for pain surveys and joint features. |
| `apply_pca` | `training_data: pd.DataFrame`<br>`test_data: pd.DataFrame` | `tuple[pd.DataFrame, pd.DataFrame]` | Apply PCA (Proper Orthogonal Decomposition) to temporal dimension of each feature, preserving explained variance ≥0.95. Returns POD global features. |
| `apply_feature_selection` | `train_global_features: pd.DataFrame`<br>`test_global_features: pd.DataFrame`<br>`train_labels: pd.Series` | `tuple[pd.DataFrame, pd.DataFrame]` | Supervised feature selection on global features using variance thresholding, correlation filtering, Random Forest, and Mutual Information. |
| `computeAndSaveClassWeights` | `labels: pd.DataFrame`<br>`savingPath: str` | - | Compute inverse frequency class weights (N / (C × Nc)) and save to YAML file. |
| `preprocess` | - | - | **Main entry point**: orchestrates full preprocessing pipeline including loading, feature engineering, normalization, PCA, feature selection, and saving results. |
| `__aggregate_time_series` | `data: pd.DataFrame`<br>`primaryKeyColumn: str`<br>`timeColumn: str`<br>`extraColumns: list[str]` | `np.ndarray` | Private method to pivot time series into 3D array (samples × timesteps × features). |
| `__plotPcaNumComponentsPerFeature` | `pcadata: list[np.ndarray]`<br>`feature_names: list[str]` | - | Private method to visualize number of PCA components retained per feature. |

---

## Processing Pipeline

The `preprocess()` method orchestrates the following steps:

1. **Load Raw Data**: Read training/test CSV files and labels from `path_raw_data`
2. **Remove Last Column**: Drop unnecessary trailing column from raw data
3. **Feature Engineering**:
   - Map categorical n_eyes ∈ {two, one+eye_patch, missing} to binary isPirate ∈ {0,1}
   - Drop redundant features (n_legs, n_hands)
   - Optional: One-hot encode isPirate → {isPirate, isNotPirate}
4. **Visualization** (if verbose=True): Plot sample time series for inspection
5. **Normalization**: Z-score normalization per feature using training statistics: x_norm = (x - μ) / σ
6. **Global Feature Extraction** (if enabled):
   - Statistical features: mean, std, min, max, range per time series column
   - Trend features: linear slope, first/last values, total change
   - Domain features: pain survey aggregations, joint activity measures
7. **PCA/POD Application** (if enabled):
   - Apply PCA to temporal dimension of each feature independently
   - Retain components preserving ≥95% explained variance
   - Output POD coefficients as additional global features
8. **Feature Selection** (if enabled):
   - Supervised selection on global features using RandomForest + MutualInformation
   - Variance thresholding and correlation filtering
   - Preserve isPirate and sample_index columns
9. **Class Weight Computation**: Calculate inverse frequency weights for balanced loss
10. **Save Outputs**:
    - Time series: `{train,test}_data.csv` (normalized, 34 features × 160 timesteps)
    - Global features: `{train,test}_global_features.csv` (one row per sample)
    - Labels: `train_labels.csv`
    - Class weights: `class_weights.yaml`

---

## Input Data Format

### Training Data CSV
```csv
sample_index,time,n_legs,n_hands,n_eyes,pain_survey_1,pain_survey_2,...
0,0.0,2,2,two,3.5,2.1,...
0,1.0,2,2,two,3.6,2.2,...
1,0.0,1,1,one+eye_patch,1.2,0.8,...
```

### Training Labels CSV
```csv
sample_index,label
0,no_pain
1,low_pain
2,high_pain
```

---

## Output Data Format

### Processed Time Series
**Files**: `pirate_pain_train.csv`, `pirate_pain_test.csv`
```csv
sample_index,time,pain_survey_1,pain_survey_2,...,joint_29
0,0.0,0.123,-0.456,...,1.234
0,1.0,0.125,-0.450,...,1.240
...
```
- Shape: (N × 160) rows, 34 feature columns (normalized)
- isPirate column removed (moved to global features)

### Global Features
**Files**: `train_global_features.csv`, `test_global_features.csv`
```csv
sample_index,isPirate,pain_survey_1_mean,pain_survey_1_std,...,POD_0,POD_1,...
0,0,0.123,0.456,...,1.234,0.789,...
1,1,-0.234,0.567,...,-0.456,0.123,...
...
```
- Shape: N rows (one per sample), variable columns depending on configuration
- Always includes: sample_index, isPirate
- Optional: statistical features, trend features, domain features, POD coefficients
- If feature selection enabled: reduced to top-k features

### Class Weights
**File**: `class_weights.yaml`
```yaml
0: 1.2345  # no_pain weight
1: 0.8765  # low_pain weight  
2: 1.5432  # high_pain weight
```
- Used for weighted cross-entropy loss during training
- Formula: w_c = N / (C × N_c)

---

## Example Configuration

```yaml
# Data paths
path_raw_data: "../dataset/Pirate"
path_processed_data: "../dataset/PirateProcessed"

# File names
name_train_file: "pirate_pain_train.csv"
name_test_file: "pirate_pain_test.csv"
name_train_labels_file: "pirate_pain_train_labels.csv"

# Feature engineering
drop_all_is_pirate: false
one_hot_encode: false

# Global feature extraction
extract_global_features: true
global_features_to_extract: ["statistical", "trend", "domain"]

# PCA/POD
PCA: true
explained_variance: 0.95

# Feature selection
feature_selection: true
feature_selection_params:
  method: "all_three_intersection"
  variance_threshold: 0.01
  correlation_threshold: 0.95
  top_k_rf: 300
  top_k_mi: 300
  random_state: 42
  selected_features_file: "selected_features.txt"

# Visualization
verbose: true
```

---

## Integration with Pipeline

The preprocessed data is consumed by the DataLoader module:
- Time series data → (batch, features=34, seq_len=160) tensors for encoders
- Global features → (batch, d_g) tensors for feedforward networks
- Class weights → loaded for weighted loss functions
- Sample_index preserved for window aggregation and submission generation


