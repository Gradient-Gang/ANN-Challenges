# PreProcessing Module

## Description
The PreProcessing module provides a comprehensive data preprocessing pipeline for the pirate pain dataset. It handles data loading, feature engineering, normalization, dimensionality reduction, and visualization. The module is configured via YAML files for reproducible preprocessing workflows.

## Main Class: `PreProcessor`

### Configuration Parameters
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

### Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `fromYaml(path: str)` | `path` (str): Path to YAML configuration file | `PreProcessor`: Instance initialized with YAML parameters | Builds a PreProcessor from a YAML file. |
| `__init__(self, params: dict)` | `params` (dict): Dictionary containing configuration parameters | - | Initialize the PreProcessor with given parameters. |
| `load_data(self, file_name: str)` | `file_name` (str): Name of CSV file (in path_raw_data directory) | `pd.DataFrame`: Loaded data | Load data from a CSV file. |
| `save_data(self, data, file_name: str)` | `data` (pd.DataFrame or np.ndarray): Data to save<br>`file_name` (str): Name of output CSV file | - | Save data to a CSV file. |
| `remove_last_column(self, data: pd.DataFrame)` | `data` (pd.DataFrame): Input data | `pd.DataFrame`: Data with last column removed | Remove the last column from the DataFrame. |
| `handle_is_pirate_features(self, data: pd.DataFrame)` | `data` (pd.DataFrame): Input data with pirate features | `pd.DataFrame`: Processed data | Handle isPirate features by dropping or encoding them. |
| `normalize_per_process(self, training_data: pd.DataFrame, test_data: pd.DataFrame)` | `training_data` (pd.DataFrame): Training data<br>`test_data` (pd.DataFrame): Test data | Normalized training and test data | Normalize numerical features to have zero mean and unit variance. |
| `plot_one_time_series(self, data: pd.DataFrame, number: int)` | `data` (pd.DataFrame): Data with time series features<br>`number` (int): Number of samples to plot | - | Plot a single time series from the DataFrame. |
| `__aggregate_time_series(self, data: pd.DataFrame, primaryKeyColumn: str, timeColumn: str, extraColumns: list[str])` | `data` (pd.DataFrame): Input data<br>`primaryKeyColumn` (str): Column name for primary key<br>`timeColumn` (str): Column name for time<br>`extraColumns` (list[str]): Additional columns to exclude | `np.ndarray`: 3D array of aggregated time series | Aggregate time series data into a 3D NumPy array. |
| `__plotPcaNumComponentsPerFeature(self, pcadata: list[np.ndarray], feature_names: list[str])` | `pcadata` (list[np.ndarray]): List of PCA-transformed data per feature<br>`feature_names` (list[str]): Names of features | - | Plot the number of PCA components selected per feature. |
| `apply_pca(self, training_data: pd.DataFrame, test_data: pd.DataFrame)` | `training_data` (pd.DataFrame): Training data<br>`test_data` (pd.DataFrame): Test data | PCA-transformed training and test data | Apply PCA to reduce dimensionality of the data. |
| `apply_feature_selection(self, training_data: pd.DataFrame, test_data: pd.DataFrame)` | `training_data` (pd.DataFrame): Training data<br>`test_data` (pd.DataFrame): Test data | Feature-selected training and test data | Select specific features from the data. |
| `preprocess()` | - | - | Main preprocessing function to load, process, and save data. |

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
