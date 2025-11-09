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
| `fromYaml` | `path: str` | `PreProcessor` | Builds a PreProcessor from a YAML file. |
| `__init__` | `params: dict` | - | Initialize the PreProcessor with given parameters. |
| `load_data` | `file_name: str` | `pd.DataFrame` | Load data from a CSV file. |
| `save_data` | `data`<br>`file_name: str` | - | Save data to a CSV file. |
| `remove_last_column` | `data: pd.DataFrame` | `pd.DataFrame` | Remove the last column from the DataFrame. |
| `handle_is_pirate_features` | `data: pd.DataFrame` | `pd.DataFrame` | Handle isPirate features by dropping or encoding them. |
| `normalize_per_process` | `training_data: pd.DataFrame`<br>`test_data: pd.DataFrame` | Normalized data | Normalize numerical features to have zero mean and unit variance. |
| `plot_one_time_series` | `data: pd.DataFrame`<br>`number: int` | - | Plot a single time series from the DataFrame. |
| `__aggregate_time_series` | `data: pd.DataFrame`<br>`primaryKeyColumn: str`<br>`timeColumn: str`<br>`extraColumns: list[str]` | `np.ndarray` | Aggregate time series data into a 3D NumPy array. |
| `__plotPcaNumComponentsPerFeature` | `pcadata: list[np.ndarray]`<br>`feature_names: list[str]` | - | Plot the number of PCA components selected per feature. |
| `apply_pca` | `training_data: pd.DataFrame`<br>`test_data: pd.DataFrame` | PCA-transformed data | Apply PCA to reduce dimensionality of the data. |
| `apply_feature_selection` | `training_data: pd.DataFrame`<br>`test_data: pd.DataFrame` | Feature-selected data | Select specific features from the data. |
| `preprocess` | - | - | Main preprocessing function to load, process, and save data. |

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


