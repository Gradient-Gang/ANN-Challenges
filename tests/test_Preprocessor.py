import pytest
import os
import tempfile
import pandas as pd
import yaml
from GradientGang.PreProcessing.PreProcessor import PreProcessor


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML config file."""
    config = {
        "path_raw_data": temp_dir,
        "path_processed_data": temp_dir,
        "name_train_file": "train.csv",
        "name_test_file": "test.csv",
        "name_train_labels_file": "train_labels.csv"
    }
    
    yaml_path = os.path.join(temp_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    
    return yaml_path


@pytest.fixture
def sample_params():
    """Create sample parameters dictionary."""
    return {
        "path_raw_data": "dataset/raw",
        "path_processed_data": "dataset/processed",
        "name_train_file": "train.csv",
        "name_test_file": "test.csv",
        "name_train_labels_file": "train_labels.csv"
    }


class TestPreProcessorInitialization:
    """Test suite for PreProcessor initialization."""

    def test_initialization_with_params(self, sample_params):
        """Test initialization with parameters dictionary."""
        preprocessor = PreProcessor(sample_params)
        
        assert preprocessor.params == sample_params
        assert preprocessor.path_raw_data == "dataset/raw"
        assert preprocessor.path_processed_data == "dataset/processed"
        assert preprocessor.name_train_file == "train.csv"
        assert preprocessor.name_test_file == "test.csv"

    def test_initialization_with_defaults(self):
        """Test initialization with minimal parameters."""
        params = {}
        preprocessor = PreProcessor(params)
        
        assert preprocessor.path_raw_data == ""
        assert preprocessor.path_processed_data == ""
        assert preprocessor.name_train_file == "train.csv"
        assert preprocessor.name_test_file == "test.csv"

    def test_from_yaml(self, sample_yaml_config):
        """Test creating PreProcessor from YAML file."""
        preprocessor = PreProcessor.fromYAML(sample_yaml_config)
        
        assert isinstance(preprocessor, PreProcessor)
        assert preprocessor.name_train_file == "train.csv"
        assert preprocessor.name_test_file == "test.csv"


class TestPreProcessorBasicFunctionality:
    """Test suite for basic PreProcessor functionality."""

    def test_preprocessor_stores_params(self, sample_params):
        """Test that PreProcessor stores parameters correctly."""
        preprocessor = PreProcessor(sample_params)
        
        assert "path_raw_data" in preprocessor.params
        assert "path_processed_data" in preprocessor.params
        assert preprocessor.params["name_train_file"] == "train.csv"

    def test_preprocessor_params_immutable(self, sample_params):
        """Test that stored params can be accessed."""
        preprocessor = PreProcessor(sample_params)
        
        # Should be able to access params
        assert preprocessor.params.get("path_raw_data") is not None


class TestPreProcessorDataOperations:
    """Test suite for PreProcessor data operations."""

    def test_load_data(self, temp_dir):
        """Test loading data from CSV."""
        # Create sample CSV file
        sample_data = pd.DataFrame({
            'sample_index': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0]
        })
        csv_path = os.path.join(temp_dir, 'data.csv')
        sample_data.to_csv(csv_path, index=False)
        
        # Create preprocessor
        params = {'path_raw_data': temp_dir}
        preprocessor = PreProcessor(params)
        
        # Load data
        loaded_data = preprocessor.load_data('data.csv')
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 4
        assert 'feature1' in loaded_data.columns

    def test_save_data_dataframe(self, temp_dir):
        """Test saving DataFrame to CSV."""
        params = {'path_processed_data': temp_dir}
        preprocessor = PreProcessor(params)
        
        # Create sample data
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        # Save data
        preprocessor.save_data(data, 'output.csv')
        
        # Verify file exists
        output_path = os.path.join(temp_dir, 'output.csv')
        assert os.path.exists(output_path)
        
        # Verify content
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == ['col1', 'col2']

    def test_save_data_numpy_array(self, temp_dir):
        """Test saving NumPy array to CSV."""
        params = {'path_processed_data': temp_dir}
        preprocessor = PreProcessor(params)
        
        # Create sample numpy array
        import numpy as np
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Save data
        preprocessor.save_data(data, 'numpy_output.csv')
        
        # Verify file exists
        output_path = os.path.join(temp_dir, 'numpy_output.csv')
        assert os.path.exists(output_path)

    def test_remove_last_column(self):
        """Test removing last column from DataFrame."""
        params = {}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        
        result = preprocessor.remove_last_column(data)
        
        assert len(result.columns) == 2
        assert 'col1' in result.columns
        assert 'col2' in result.columns
        assert 'col3' not in result.columns


class TestPreProcessorFeatureHandling:
    """Test suite for PreProcessor feature handling."""

    def test_handle_is_pirate_features_default(self):
        """Test default isPirate feature handling."""
        params = {}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'n_legs': [2, 1, 2],
            'n_hands': [2, 1, 2],
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'feature1': [1.0, 2.0, 3.0]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        # n_legs and n_hands should be dropped
        assert 'n_legs' not in result.columns
        assert 'n_hands' not in result.columns
        assert 'n_eyes' not in result.columns
        
        # isPirate should be created
        assert 'isPirate' in result.columns
        assert result['isPirate'].iloc[0] == 0  # two eyes -> 0
        assert result['isPirate'].iloc[1] == 1  # one+eye_patch -> 1

    def test_handle_is_pirate_features_drop_all(self):
        """Test dropping all isPirate features."""
        params = {'drop_all_is_pirate': True}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'n_legs': [2, 1, 2],
            'n_hands': [2, 1, 2],
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'feature1': [1.0, 2.0, 3.0]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        # All pirate-related columns should be dropped
        assert 'n_legs' not in result.columns
        assert 'n_hands' not in result.columns
        assert 'n_eyes' not in result.columns

    def test_handle_is_pirate_features_one_hot_encode(self):
        """Test one-hot encoding isPirate feature."""
        params = {'one_hot_encode': True}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'n_legs': [2, 1, 2],
            'n_hands': [2, 1, 2],
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'feature1': [1.0, 2.0, 3.0]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        # isPirate and isNotPirate should be created
        assert 'isPirate' in result.columns
        assert 'isNotPirate' in result.columns
        
        # Check complementary relationship
        assert result['isPirate'].iloc[0] + result['isNotPirate'].iloc[0] == 1
        assert result['isPirate'].iloc[1] + result['isNotPirate'].iloc[1] == 1


class TestPreProcessorNormalization:
    """Test suite for PreProcessor normalization."""

    def test_normalize_per_process_basic(self):
        """Test basic normalization."""
        params = {}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [0, 1, 2, 3],
            'time': [0, 1, 0, 1],
            'feature1': [10.0, 20.0, 30.0, 40.0],
            'feature2': [100.0, 200.0, 300.0, 400.0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 1],
            'time': [0, 1],
            'feature1': [15.0, 25.0],
            'feature2': [150.0, 250.0]
        })
        
        train_normalized, test_normalized = preprocessor.normalize_per_process(
            train_data.copy(), test_data.copy()
        )
        
        # sample_index and time should not be normalized
        assert train_normalized['sample_index'].equals(train_data['sample_index'])
        assert train_normalized['time'].equals(train_data['time'])
        
        # feature1 should be normalized (mean=25, std~12.9)
        assert abs(train_normalized['feature1'].mean()) < 1e-10
        assert abs(train_normalized['feature1'].std() - 1.0) < 1e-10

    def test_normalize_per_process_zero_std(self):
        """Test normalization with zero standard deviation."""
        params = {}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'feature1': [5.0, 5.0, 5.0],  # Constant feature
            'feature2': [10.0, 20.0, 30.0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 1],
            'feature1': [5.0, 5.0],
            'feature2': [15.0, 25.0]
        })
        
        train_normalized, test_normalized = preprocessor.normalize_per_process(
            train_data.copy(), test_data.copy()
        )
        
        # Constant feature should be centered but not scaled
        assert abs(train_normalized['feature1'].mean()) < 1e-10
        assert all(train_normalized['feature1'] == 0)

    def test_normalize_per_process_excluded_columns(self):
        """Test that excluded columns are not normalized."""
        params = {'columns_excluded_from_normalization': ['sample_index', 'time', 'isPirate']}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'time': [0, 1, 2],
            'isPirate': [0, 1, 0],
            'feature1': [10.0, 20.0, 30.0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 1],
            'time': [0, 1],
            'isPirate': [1, 0],
            'feature1': [15.0, 25.0]
        })
        
        train_normalized, test_normalized = preprocessor.normalize_per_process(
            train_data.copy(), test_data.copy()
        )
        
        # Excluded columns should remain unchanged
        assert train_normalized['sample_index'].equals(train_data['sample_index'])
        assert train_normalized['isPirate'].equals(train_data['isPirate'])
        
        # feature1 should be normalized
        assert abs(train_normalized['feature1'].mean()) < 1e-10


class TestPreProcessorConfiguration:
    """Test suite for PreProcessor configuration options."""

    def test_pca_configuration(self):
        """Test PCA configuration."""
        params = {
            'PCA': True,
            'explained_variance': 0.99
        }
        preprocessor = PreProcessor(params)
        
        assert preprocessor.use_pca is True
        assert preprocessor.explained_variance == 0.99

    def test_feature_selection_configuration(self):
        """Test feature selection configuration."""
        params = {
            'feature_selection': True,
            'feature_selected': ['feature1', 'feature2']
        }
        preprocessor = PreProcessor(params)
        
        assert preprocessor.use_feature_selection is True
        assert preprocessor.feature_selected == ['feature1', 'feature2']

    def test_verbose_configuration(self):
        """Test verbose configuration."""
        params = {'verbose': False}
        preprocessor = PreProcessor(params)
        
        assert preprocessor.verbose is False

    def test_default_verbose(self):
        """Test default verbose is True."""
        params = {}
        preprocessor = PreProcessor(params)
        
        assert preprocessor.verbose is True


class TestPreProcessorFeatureSelection:
    """Test suite for PreProcessor feature selection."""

    def test_apply_feature_selection(self):
        """Test feature selection."""
        params = {
            'feature_selection': True,
            'feature_selected': ['sample_index', 'feature1']
        }
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': [7.0, 8.0, 9.0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 1],
            'feature1': [10.0, 11.0],
            'feature2': [12.0, 13.0],
            'feature3': [14.0, 15.0]
        })
        
        train_selected, test_selected = preprocessor.apply_feature_selection(
            train_data, test_data
        )
        
        # Only selected features should remain
        assert list(train_selected.columns) == ['sample_index', 'feature1']
        assert list(test_selected.columns) == ['sample_index', 'feature1']
        assert len(train_selected) == 3
        assert len(test_selected) == 2


class TestPreProcessorClassWeights:
    """Test suite for PreProcessor class weights computation."""

    def test_compute_and_save_class_weights(self, temp_dir):
        """Test computing and saving class weights."""
        params = {}
        preprocessor = PreProcessor(params)
        
        # Create sample labels with imbalanced classes
        labels = pd.DataFrame({
            'sample_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'label': ['no_pain', 'no_pain', 'no_pain', 'no_pain', 'no_pain',
                     'low_pain', 'low_pain', 'low_pain', 'high_pain', 'high_pain']
        })
        
        save_path = os.path.join(temp_dir, 'class_weights.yaml')
        preprocessor.computeAndSaveClassWeights(labels, savingPath=save_path)
        
        # Verify file exists
        assert os.path.exists(save_path)
        
        # Load and verify weights
        with open(save_path, 'r') as f:
            weights = yaml.safe_load(f)
        
        # Should have 3 classes (0: no_pain, 1: low_pain, 2: high_pain)
        assert 0 in weights
        assert 1 in weights
        assert 2 in weights
        
        # no_pain (5 samples) should have lower weight than high_pain (2 samples)
        assert weights[0] < weights[2]

    def test_compute_class_weights_balanced(self, temp_dir):
        """Test computing class weights with balanced classes."""
        params = {}
        preprocessor = PreProcessor(params)
        
        # Create sample labels with balanced classes
        labels = pd.DataFrame({
            'sample_index': [0, 1, 2, 3, 4, 5],
            'label': ['no_pain', 'no_pain', 'low_pain', 'low_pain', 'high_pain', 'high_pain']
        })
        
        save_path = os.path.join(temp_dir, 'balanced_weights.yaml')
        preprocessor.computeAndSaveClassWeights(labels, savingPath=save_path)
        
        # Load weights
        with open(save_path, 'r') as f:
            weights = yaml.safe_load(f)
        
        # All weights should be equal (or very close) for balanced classes
        assert abs(weights[0] - weights[1]) < 1e-10
        assert abs(weights[1] - weights[2]) < 1e-10
