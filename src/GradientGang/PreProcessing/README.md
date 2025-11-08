# PreProcessing Module

## Description
The PreProcessing module provides a comprehensive data preprocessing pipeline for the pirate pain dataset. It handles data loading, feature engineering, normalization, dimensionality reduction, and visualization. The module is configured via YAML files for reproducible preprocessing workflows.

## Main Class: `PreProcessing`

### Methods

#### `__init__(self, path_params: str)`
Initialize the PreProcessing class by loading parameters from a YAML file.

**Parameters:**
- `path_params` (str): Path to YAML configuration file

**Configuration Parameters:**
- `path_raw_data` (str): Directory containing raw CSV files
- `path_processed_data` (str): Directory for saving processed files
- `name_train_file` (str): Training data filename (default: "train.csv")
- `name_test_file` (str): Test data filename (default: "test.csv")
- `name_train_labels_file` (str): Training labels filename (default: "train_labels.csv")
- `drop_all_is_pirate` (bool): Whether to drop all pirate-related features (default: False)
- `one_hot_encode` (bool): Whether to one-hot encode n_eyes feature (default: False)
- `pca` (bool): Whether to apply PCA (default: False)
- `explained_variance` (float): PCA explained variance ratio (default: 0.95)
- `feature_selection` (bool): Whether to apply feature selection (default: False)
- `feature_selected` (list): List of features to select
- `verbose` (bool): Whether to display plots (default: True)

---

#### `load_data(self, file_name: str) -> pd.DataFrame`
Load data from a CSV file.

**Parameters:**
- `file_name` (str): Name of CSV file (in path_raw_data directory)

**Returns:**
- `pd.DataFrame`: Loaded data

---

#### `save_data(self, data, file_name: str)`
Save data to a CSV file.

**Parameters:**
- `data` (pd.DataFrame or np.ndarray): Data to save
- `file_name` (str): Name of output CSV file (in path_processed_data directory)

**Note:** Automatically converts numpy arrays to DataFrames

---

#### `remove_last_column(self, data: pd.DataFrame) -> pd.DataFrame`
Remove the last column from the DataFrame.

**Parameters:**
- `data` (pd.DataFrame): Input data

**Returns:**
- `pd.DataFrame`: Data with last column removed

---

#### `handle_inspirate_features(self, data: pd.DataFrame) -> pd.DataFrame`
Handle pirate-specific features (n_legs, n_hands, n_eyes).

**Parameters:**
- `data` (pd.DataFrame): Input data with pirate features

**Returns:**
- `pd.DataFrame`: Processed data

**Behavior:**
- If `drop_all_is_pirate=True`: Drops n_legs, n_hands, n_eyes
- If `one_hot_encode=True`: One-hot encodes n_eyes, drops n_legs and n_hands
- Default: Maps n_eyes to numeric ("two"→2, "one+eye_patch"→1), drops other pirate features

---

#### `normalize_per_process(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
Normalize numerical features to zero mean and unit variance.

**Parameters:**
- `training_data` (pd.DataFrame): Training data
- `test_data` (pd.DataFrame): Test data

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: (normalized_train, normalized_test)

**Behavior:**
- Uses training statistics (mean, std) for both train and test
- Handles constant features (zero std) by centering only
- Processes only numeric columns

---

#### `plot_one_time_series(self, data: pd.DataFrame, number: int)`
Plot time series visualizations for selected samples.

**Parameters:**
- `data` (pd.DataFrame): Data with time series features
- `number` (int): Number of samples to plot

**Plots:**
- pain_survey_1, pain_survey_2, pain_survey_3, pain_survey_4
- joint_00, joint_01, joint_28, joint_29

---

#### `apply_pca(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`
Apply PCA for dimensionality reduction.

**Parameters:**
- `training_data` (pd.DataFrame): Training data
- `test_data` (pd.DataFrame): Test data

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (train_pca, test_pca)

**Behavior:**
- Fits PCA on training data
- Transforms both train and test
- Uses explained_variance parameter from config

---

#### `apply_feature_selection(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
Select specific features from the data.

**Parameters:**
- `training_data` (pd.DataFrame): Training data
- `test_data` (pd.DataFrame): Test data

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: (selected_train, selected_test)

**Raises:**
- `ValueError`: If feature_selected is not set in config

---

#### `preprocess(self)`
Main preprocessing function to load, process, and save data.

**Workflow:**
1. Load train, test, and label files
2. Remove last column
3. Handle pirate features
4. Plot time series (if verbose=True)
5. Normalize data
6. Plot normalized time series (if verbose=True)
7. Apply PCA (if enabled)
8. Apply feature selection (if enabled)
9. Save processed data

**Error Handling:**
- Prints error messages at each step
- Continues execution when possible (e.g., plotting errors)

---

## Example Usage

### Basic Preprocessing

```python
from GradientGang.PreProcessing import PreProcessing

# Create preprocessing instance
preprocessor = PreProcessing("Notebook/Params/preprocessing_params.yaml")

# Run preprocessing pipeline
preprocessor.preprocess()
# Output:
# Data loaded successfully.
# Last column removed successfully.
# Inspirate features handled successfully.
# Data normalized successfully.
# PCA not applied.
# Feature selection not applied.
# Data saved successfully.
```

### Configuration File (YAML)

```yaml
# Paths
path_raw_data: "dataset/Pirate"
path_processed_data: "dataset/PirateProcessed"

# File names
name_train_file: "pirate_pain_train.csv"
name_test_file: "pirate_pain_test.csv"
name_train_labels_file: "pirate_pain_train_labels.csv"

# Feature handling
drop_all_is_pirate: false
one_hot_encode: false

# PCA
pca: true
explained_variance: 0.95

# Feature selection
feature_selection: false
feature_selected: null

# Verbosity
verbose: true
```

### With PCA

```yaml
pca: true
explained_variance: 0.90  # Keep 90% of variance
```

### With Feature Selection

```yaml
feature_selection: true
feature_selected:
  - pain_survey_1
  - pain_survey_2
  - joint_00
  - joint_01
  - joint_28
```

### Drop Pirate Features

```yaml
drop_all_is_pirate: true
```

### One-Hot Encode Eyes

```yaml
drop_all_is_pirate: false
one_hot_encode: true
```

---

## Processing Steps

1. **Load Data**: Read CSV files from raw data directory
2. **Remove Last Column**: Drop unnecessary trailing column
3. **Handle Pirate Features**:
   - Map n_eyes to numeric values
   - Optional: Drop all pirate features
   - Optional: One-hot encode n_eyes
4. **Visualization**: Plot sample time series (optional)
5. **Normalization**: Z-score normalization using training statistics
6. **PCA**: Dimensionality reduction (optional)
7. **Feature Selection**: Select specific features (optional)
8. **Save**: Write processed data to CSV files

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

After preprocessing, data is saved in the same CSV format but with:
- Numeric features normalized (mean=0, std=1)
- Pirate features processed as configured
- Optionally reduced dimensions (PCA)
- Optionally filtered features (feature selection)

---

## Notes

- All transformations use training statistics to avoid data leakage
- PCA and normalization fit only on training data, transform both train and test
- Visualization requires matplotlib (optional, errors are caught)
- The module gracefully handles missing columns with error messages
- Fixed random seed (42) should be used in preprocessing for reproducibility
