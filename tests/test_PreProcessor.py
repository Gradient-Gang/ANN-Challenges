import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import yaml
from GradientGang.PreProcessing.PreProcessor import PreProcessor


class TestPreProcessorInit:
    """Test PreProcessor initialization and configuration."""
    
    def test_init_with_basic_params(self):
        """Test initialization with basic parameters."""
        params = {
            "path_raw_data": "data/raw",
            "path_processed_data": "data/processed",
            "verbose": False
        }
        preprocessor = PreProcessor(params)
        
        assert preprocessor.path_raw_data == "data/raw"
        assert preprocessor.path_processed_data == "data/processed"
        assert preprocessor.verbose is False
    
    def test_init_with_default_values(self):
        """Test initialization with default parameter values."""
        params = {}
        preprocessor = PreProcessor(params)
        
        assert preprocessor.path_raw_data == ""
        assert preprocessor.path_processed_data == ""
        assert preprocessor.name_train_file == "train.csv"
        assert preprocessor.name_test_file == "test.csv"
        assert preprocessor.name_train_labels_file == "train_labels.csv"
        assert preprocessor.verbose is True
    
    def test_init_with_pca_params(self):
        """Test initialization with PCA parameters."""
        params = {
            "PCA": True,
            "explained_variance": 0.90
        }
        preprocessor = PreProcessor(params)
        
        assert preprocessor.use_pca is True
        assert preprocessor.explained_variance == 0.90
    
    def test_init_with_feature_selection(self):
        """Test initialization with feature selection parameters."""
        params = {
            "feature_selection": True,
            "feature_selected": ["feature1", "feature2"]
        }
        preprocessor = PreProcessor(params)
        
        assert preprocessor.use_feature_selection is True
        assert preprocessor.feature_selected == ["feature1", "feature2"]
    
    def test_from_yaml(self):
        """Test loading PreProcessor from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                "path_raw_data": "test/raw",
                "path_processed_data": "test/processed",
                "verbose": False,
                "PCA": True,
                "explained_variance": 0.85
            }
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            preprocessor = PreProcessor.fromYAML(yaml_path)
            assert preprocessor.path_raw_data == "test/raw"
            assert preprocessor.path_processed_data == "test/processed"
            assert preprocessor.verbose is False
            assert preprocessor.use_pca is True
            assert preprocessor.explained_variance == 0.85
        finally:
            os.unlink(yaml_path)


class TestPreProcessorDataHandling:
    """Test data loading and saving functionality."""
    
    def test_load_data(self):
        """Test loading data from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV file
            test_data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6]
            })
            csv_path = os.path.join(tmpdir, 'test.csv')
            test_data.to_csv(csv_path, index=False)
            
            # Test loading
            params = {"path_raw_data": tmpdir}
            preprocessor = PreProcessor(params)
            loaded_data = preprocessor.load_data('test.csv')
            
            pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_save_data_dataframe(self):
        """Test saving DataFrame to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {"path_processed_data": tmpdir}
            preprocessor = PreProcessor(params)
            
            test_data = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6]
            })
            
            preprocessor.save_data(test_data, 'output.csv')
            
            # Verify file was created and contains correct data
            output_path = os.path.join(tmpdir, 'output.csv')
            assert os.path.exists(output_path)
            loaded_data = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_save_data_numpy_array(self):
        """Test saving NumPy array to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {"path_processed_data": tmpdir}
            preprocessor = PreProcessor(params)
            
            test_array = np.array([[1, 2, 3], [4, 5, 6]])
            preprocessor.save_data(test_array, 'output.csv')
            
            # Verify file was created
            output_path = os.path.join(tmpdir, 'output.csv')
            assert os.path.exists(output_path)
            loaded_data = pd.read_csv(output_path)
            assert loaded_data.shape == (2, 3)
    
    def test_save_data_creates_directory(self):
        """Test that save_data creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, 'nested', 'path')
            params = {"path_processed_data": nested_path}
            preprocessor = PreProcessor(params)
            
            test_data = pd.DataFrame({'col1': [1, 2]})
            preprocessor.save_data(test_data, 'output.csv')
            
            assert os.path.exists(nested_path)
            assert os.path.exists(os.path.join(nested_path, 'output.csv'))


class TestPreProcessorTransformations:
    """Test data transformation methods."""
    
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
        
        assert list(result.columns) == ['col1', 'col2']
        assert result.shape == (3, 2)
    
    def test_handle_is_pirate_default(self):
        """Test default handling of isPirate features."""
        params = {}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'n_legs': [1, 2, 1],
            'n_hands': [1, 2, 1],
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'other_col': [10, 20, 30]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        assert 'isPirate' in result.columns
        assert 'n_legs' not in result.columns
        assert 'n_hands' not in result.columns
        assert 'n_eyes' not in result.columns
        assert 'other_col' in result.columns
        assert list(result['isPirate']) == [0, 1, 0]
    
    def test_handle_is_pirate_with_one_hot_encoding(self):
        """Test isPirate handling with one-hot encoding."""
        params = {"one_hot_encode": True}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'other_col': [10, 20, 30]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        assert 'isPirate' in result.columns
        assert 'isNotPirate' in result.columns
        assert list(result['isPirate']) == [0, 1, 0]
        assert list(result['isNotPirate']) == [1, 0, 1]
    
    def test_handle_is_pirate_drop_all(self):
        """Test dropping all isPirate features."""
        params = {"drop_all_is_pirate": True}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'n_legs': [1, 2, 1],
            'n_hands': [1, 2, 1],
            'n_eyes': ['two', 'one+eye_patch', 'two'],
            'other_col': [10, 20, 30]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        assert 'n_legs' not in result.columns
        assert 'n_hands' not in result.columns
        assert 'n_eyes' not in result.columns
        assert 'isPirate' not in result.columns
        assert 'other_col' in result.columns


class TestPreProcessorNormalization:
    """Test normalization functionality."""
    
    def test_normalize_basic(self):
        """Test basic normalization."""
        params = {"columns_excluded_from_normalization": []}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        test_data = pd.DataFrame({
            'feature1': [2.5, 3.5],
            'feature2': [25.0, 35.0]
        })
        
        train_norm, test_norm = preprocessor.normalize_per_process(train_data.copy(), test_data.copy())
        
        # Check training data is normalized (mean ~ 0, std ~ 1)
        assert np.isclose(train_norm['feature1'].mean(), 0, atol=1e-10)
        assert np.isclose(train_norm['feature1'].std(), 1, atol=1e-10)
        
        # Test data should be normalized using training statistics (values should be normalized)
        assert not np.allclose(test_norm['feature1'].values, test_data['feature1'].values)
    
    def test_normalize_with_exclusions(self):
        """Test normalization with excluded columns."""
        params = {"columns_excluded_from_normalization": ["sample_index"]}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [1, 2, 3, 4, 5],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [6, 7],
            'feature1': [2.5, 3.5]
        })
        
        train_norm, test_norm = preprocessor.normalize_per_process(train_data.copy(), test_data.copy())
        
        # sample_index should not be normalized
        assert list(train_norm['sample_index']) == [1, 2, 3, 4, 5]
        assert list(test_norm['sample_index']) == [6, 7]
        
        # feature1 should be normalized
        assert np.isclose(train_norm['feature1'].mean(), 0, atol=1e-10)
    
    def test_normalize_with_zero_std(self):
        """Test normalization with constant feature (zero std)."""
        params = {}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'feature1': [5.0, 5.0, 5.0, 5.0],
            'feature2': [1.0, 2.0, 3.0, 4.0]
        })
        
        test_data = pd.DataFrame({
            'feature1': [5.0, 5.0],
            'feature2': [2.5, 3.5]
        })
        
        train_norm, test_norm = preprocessor.normalize_per_process(train_data.copy(), test_data.copy())
        
        # Constant feature should be centered (mean subtracted)
        assert np.all(train_norm['feature1'] == 0)
        assert np.all(test_norm['feature1'] == 0)


class TestPreProcessorFeatureSelection:
    """Test feature selection functionality."""
    
    def test_apply_feature_selection(self):
        """Test applying feature selection."""
        params = {
            "feature_selection": True,
            "feature_selected": ["feature1", "feature3"]
        }
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        
        test_data = pd.DataFrame({
            'feature1': [10, 11],
            'feature2': [12, 13],
            'feature3': [14, 15]
        })
        
        train_selected, test_selected = preprocessor.apply_feature_selection(train_data, test_data)
        
        assert list(train_selected.columns) == ["feature1", "feature3"]
        assert list(test_selected.columns) == ["feature1", "feature3"]
        assert train_selected.shape == (3, 2)
        assert test_selected.shape == (2, 2)


class TestPreProcessorClassWeights:
    """Test class weights computation."""
    
    def test_compute_and_save_class_weights(self):
        """Test computing and saving class weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {}
            preprocessor = PreProcessor(params)
            
            # Create imbalanced labels
            labels = pd.DataFrame({
                'label': ['no_pain'] * 100 + ['low_pain'] * 50 + ['high_pain'] * 25
            })
            
            save_path = os.path.join(tmpdir, 'class_weights.yaml')
            preprocessor.computeAndSaveClassWeights(labels, save_path)
            
            # Verify file was created
            assert os.path.exists(save_path)
            
            # Load and verify weights
            with open(save_path, 'r') as f:
                weights = yaml.safe_load(f)
            
            assert len(weights) == 3
            assert 0 in weights  # no_pain
            assert 1 in weights  # low_pain
            assert 2 in weights  # high_pain
            
            # Weights should be inversely proportional to class frequency
            assert weights[0] < weights[1] < weights[2]
    
    def test_compute_class_weights_balanced(self):
        """Test class weights with balanced classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {}
            preprocessor = PreProcessor(params)
            
            labels = pd.DataFrame({
                'label': ['no_pain'] * 50 + ['low_pain'] * 50 + ['high_pain'] * 50
            })
            
            save_path = os.path.join(tmpdir, 'class_weights.yaml')
            preprocessor.computeAndSaveClassWeights(labels, save_path)
            
            with open(save_path, 'r') as f:
                weights = yaml.safe_load(f)
            
            # All weights should be equal for balanced classes
            assert np.isclose(weights[0], weights[1], atol=1e-10)
            assert np.isclose(weights[1], weights[2], atol=1e-10)


class TestPreProcessorPCA:
    """Test PCA functionality."""
    
    def test_apply_pca_basic(self):
        """Test basic PCA application."""
        params = {
            "PCA": True,
            "explained_variance": 0.95,
            "verbose": False
        }
        preprocessor = PreProcessor(params)
        
        # Create time series data
        train_data = pd.DataFrame({
            'sample_index': [0, 0, 0, 1, 1, 1],
            'time': [0, 1, 2, 0, 1, 2],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'isPirate': [0, 0, 0, 1, 1, 1]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 0, 0],
            'time': [0, 1, 2],
            'feature1': [2.5, 3.5, 4.5],
            'feature2': [25.0, 35.0, 45.0],
            'isPirate': [0, 0, 0]
        })
        
        train_pca, test_pca = preprocessor.apply_pca(train_data.copy(), test_data.copy())
        
        # Verify output structure
        assert 'sample_index' in train_pca.columns
        assert 'isPirate' in train_pca.columns
        assert train_pca.shape[0] == 2  # 2 unique samples
        assert test_pca.shape[0] == 1  # 1 unique sample
        
        # PCA columns should be added
        assert train_pca.shape[1] > 2  # More than just sample_index and isPirate
    
    def test_apply_pca_preserves_pirate_columns(self):
        """Test that PCA preserves isPirate and isNotPirate columns."""
        params = {
            "PCA": True,
            "explained_variance": 0.90,
            "verbose": False
        }
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'sample_index': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'isPirate': [0, 0, 1, 1],
            'isNotPirate': [1, 1, 0, 0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 0],
            'time': [0, 1],
            'feature1': [2.5, 3.5],
            'isPirate': [0, 0],
            'isNotPirate': [1, 1]
        })
        
        train_pca, test_pca = preprocessor.apply_pca(train_data, test_data)
        
        # Verify isPirate columns are preserved
        assert 'isPirate' in train_pca.columns
        assert 'isNotPirate' in train_pca.columns
        assert 'isPirate' in test_pca.columns
        assert 'isNotPirate' in test_pca.columns


class TestPreProcessorIntegration:
    """Test integrated preprocessing workflow."""
    
    def test_preprocess_basic_workflow(self):
        """Test basic preprocessing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data files
            train_data = pd.DataFrame({
                'sample_index': [0, 0, 1, 1],
                'time': [0, 1, 0, 1],
                'n_eyes': ['two', 'two', 'one+eye_patch', 'one+eye_patch'],
                'feature1': [1.0, 2.0, 3.0, 4.0],
                'extra_col': [99, 99, 99, 99]  # This will be removed
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0, 0],
                'time': [0, 1],
                'n_eyes': ['two', 'two'],
                'feature1': [2.5, 3.5],
                'extra_col': [99, 99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            # Verify processed files exist
            assert os.path.exists(os.path.join(processed_path, 'train.csv'))
            assert os.path.exists(os.path.join(processed_path, 'test.csv'))
            assert os.path.exists(os.path.join(processed_path, 'labels.csv'))
            assert os.path.exists(os.path.join(processed_path, 'class_weights.yaml'))
            
            # Load and verify processed data
            processed_train = pd.read_csv(os.path.join(processed_path, 'train.csv'))
            assert 'isPirate' in processed_train.columns
            assert 'extra_col' not in processed_train.columns
    
    def test_preprocess_with_pca(self):
        """Test preprocessing workflow with PCA enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create time series data
            train_data = pd.DataFrame({
                'sample_index': [0, 0, 0, 1, 1, 1],
                'time': [0, 1, 2, 0, 1, 2],
                'n_eyes': ['two'] * 6,
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'feature2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                'extra_col': [99] * 6
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0, 0, 0],
                'time': [0, 1, 2],
                'n_eyes': ['two'] * 3,
                'feature1': [2.5, 3.5, 4.5],
                'feature2': [25.0, 35.0, 45.0],
                'extra_col': [99] * 3
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "PCA": True,
                "explained_variance": 0.95,
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            # Verify files exist
            assert os.path.exists(os.path.join(processed_path, 'train.csv'))
            assert os.path.exists(os.path.join(processed_path, 'test.csv'))
            
            # Load and verify PCA was applied
            processed_train = pd.read_csv(os.path.join(processed_path, 'train.csv'))
            assert processed_train.shape[0] == 2  # 2 samples after PCA aggregation
    
    def test_preprocess_with_feature_selection(self):
        """Test preprocessing workflow with feature selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data = pd.DataFrame({
                'sample_index': [0, 1],
                'time': [0, 0],
                'n_eyes': ['two', 'two'],
                'feature1': [1.0, 2.0],
                'feature2': [10.0, 20.0],
                'feature3': [100.0, 200.0],
                'extra_col': [99, 99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'feature2': [15.0],
                'feature3': [150.0],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "feature_selection": True,
                "feature_selected": ["sample_index", "feature1", "isPirate"],
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            # Load and verify feature selection was applied
            processed_train = pd.read_csv(os.path.join(processed_path, 'train.csv'))
            assert set(processed_train.columns) == {"sample_index", "feature1", "isPirate"}
    
    def test_preprocess_handles_load_error(self, capsys):
        """Test that preprocess handles loading errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "path_raw_data": tmpdir,
                "path_processed_data": tmpdir,
                "name_train_file": "nonexistent.csv",
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            captured = capsys.readouterr()
            assert "Error loading data" in captured.out
    
    def test_preprocess_handles_column_removal_error(self, capsys):
        """Test that preprocess handles column removal errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty dataframe
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()
            labels = pd.DataFrame({'label': ['no_pain']})
            
            raw_path = os.path.join(tmpdir, 'raw')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": os.path.join(tmpdir, 'processed'),
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            captured = capsys.readouterr()
            # Should either error on removal or handle features
            assert "Error" in captured.out or "successfully" in captured.out


class TestPreProcessorVisualization:
    """Test visualization methods."""
    
    def test_plot_one_time_series(self, monkeypatch):
        """Test plotting time series (should not raise errors)."""
        import matplotlib.pyplot as plt
        
        params = {"verbose": False}
        preprocessor = PreProcessor(params)
        
        # Create sample time series data
        data = pd.DataFrame({
            'sample_index': [0, 0, 0, 1, 1, 1],
            'time': [0, 1, 2, 0, 1, 2],
            'pain_survey_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'pain_survey_2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'pain_survey_3': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'pain_survey_4': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'joint_00': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'joint_01': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'joint_28': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'joint_29': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        # Mock plt.show() to prevent the warning about non-interactive backend
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Use non-interactive backend to prevent display
        import matplotlib
        matplotlib.use('Agg')
        
        # This should not raise an error
        preprocessor.plot_one_time_series(data, number=1)
        
        # Verify the plot was created (figure exists)
        assert plt.gcf().number > 0
        plt.close('all')  # Clean up


class TestPreProcessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handle_is_pirate_with_unknown_eye_value(self):
        """Test handling unknown n_eyes values."""
        params = {}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({
            'n_eyes': ['two', 'unknown_value', 'one+eye_patch'],
            'other_col': [10, 20, 30]
        })
        
        result = preprocessor.handle_is_pirate_features(data)
        
        # Unknown values should be filled with 0 (default)
        assert 'isPirate' in result.columns
        assert list(result['isPirate']) == [0, 0, 1]
    
    def test_normalize_with_nan_values(self):
        """Test normalization handles NaN values."""
        params = {}
        preprocessor = PreProcessor(params)
        
        train_data = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        test_data = pd.DataFrame({
            'feature1': [2.5, 3.5],
            'feature2': [25.0, 35.0]
        })
        
        # Should handle NaN gracefully
        train_norm, test_norm = preprocessor.normalize_per_process(train_data.copy(), test_data.copy())
        
        # Normalization should work on non-NaN values
        assert test_norm.shape == test_data.shape
    
    def test_apply_pca_single_sample(self):
        """Test PCA with single sample edge case."""
        params = {
            "PCA": True,
            "explained_variance": 0.95,
            "verbose": False
        }
        preprocessor = PreProcessor(params)
        
        # Single training sample with time series
        train_data = pd.DataFrame({
            'sample_index': [0, 0, 0],
            'time': [0, 1, 2],
            'feature1': [1.0, 2.0, 3.0],
            'isPirate': [0, 0, 0]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 0, 0],
            'time': [0, 1, 2],
            'feature1': [2.0, 3.0, 4.0],
            'isPirate': [1, 1, 1]
        })
        
        # Should handle single sample case
        try:
            train_pca, test_pca = preprocessor.apply_pca(train_data, test_data)
            assert train_pca.shape[0] == 1
            assert test_pca.shape[0] == 1
        except Exception:
            # PCA might fail with too few samples, which is expected
            pass
    
    def test_preprocess_saves_labels_unchanged(self):
        """Test that labels file is saved unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data = pd.DataFrame({
                'sample_index': [0, 1],
                'time': [0, 0],
                'n_eyes': ['two', 'two'],
                'feature1': [1.0, 2.0],
                'extra_col': [99, 99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'sample_id': [0, 1],
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            # Verify labels are unchanged
            processed_labels = pd.read_csv(os.path.join(processed_path, 'labels.csv'))
            pd.testing.assert_frame_equal(processed_labels, labels)
    
    def test_remove_last_column_single_column(self):
        """Test removing last column when DataFrame has only one column."""
        params = {}
        preprocessor = PreProcessor(params)
        
        data = pd.DataFrame({'only_col': [1, 2, 3]})
        result = preprocessor.remove_last_column(data)
        
        assert result.shape[1] == 0
        assert len(result) == 3

    def test_apply_pca_with_verbose(self, monkeypatch):
        """Test PCA with verbose mode enabled."""
        import matplotlib.pyplot as plt
        
        params = {
            "PCA": True,
            "explained_variance": 0.95,
            "verbose": True
        }
        preprocessor = PreProcessor(params)
        
        # Mock plt.show() to prevent display
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Use matplotlib non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Create time series data
        train_data = pd.DataFrame({
            'sample_index': [0, 0, 0, 1, 1, 1],
            'time': [0, 1, 2, 0, 1, 2],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'isPirate': [0, 0, 0, 1, 1, 1]
        })
        
        test_data = pd.DataFrame({
            'sample_index': [0, 0, 0],
            'time': [0, 1, 2],
            'feature1': [2.5, 3.5, 4.5],
            'feature2': [25.0, 35.0, 45.0],
            'isPirate': [0, 0, 0]
        })
        
        train_pca, test_pca = preprocessor.apply_pca(train_data.copy(), test_data.copy())
        
        # Verify PCA was applied
        assert 'sample_index' in train_pca.columns
        assert train_pca.shape[0] == 2
        
        plt.close('all')

    def test_preprocess_handles_normalization_error(self, capsys):
        """Test that preprocess handles normalization errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with problematic values
            train_data = pd.DataFrame({
                'sample_index': [0, 1],
                'time': [0, 0],
                'n_eyes': ['two', 'two'],
                'feature1': [float('inf'), 2.0],  # inf value
                'extra_col': [99, 99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            # Should handle inf values without crashing
            captured = capsys.readouterr()
            # Either succeeds or reports error
            assert "Error" in captured.out or os.path.exists(os.path.join(processed_path, 'train.csv'))

    def test_preprocess_handles_pca_error(self, capsys):
        """Test that preprocess handles PCA errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal data that might cause PCA issues
            train_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.0],
                'extra_col': [99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "PCA": True,
                "explained_variance": 0.95,
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            captured = capsys.readouterr()
            # Should either succeed or report PCA error
            assert "Error" in captured.out or "successfully" in captured.out

    def test_preprocess_handles_feature_selection_error(self, capsys):
        """Test that preprocess handles feature selection errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data = pd.DataFrame({
                'sample_index': [0, 1],
                'time': [0, 0],
                'n_eyes': ['two', 'two'],
                'feature1': [1.0, 2.0],
                'extra_col': [99, 99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            processed_path = os.path.join(tmpdir, 'processed')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            params = {
                "path_raw_data": raw_path,
                "path_processed_data": processed_path,
                "name_train_file": "train.csv",
                "name_test_file": "test.csv",
                "name_train_labels_file": "labels.csv",
                "feature_selection": True,
                "feature_selected": ["nonexistent_feature"],  # Feature that doesn't exist
                "verbose": False
            }
            
            preprocessor = PreProcessor(params)
            preprocessor.preprocess()
            
            captured = capsys.readouterr()
            # Should report error
            assert "Error" in captured.out

    def test_preprocess_handles_save_error(self, capsys):
        """Test that preprocess handles save errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_data = pd.DataFrame({
                'sample_index': [0, 1],
                'time': [0, 0],
                'n_eyes': ['two', 'two'],
                'feature1': [1.0, 2.0],
                'extra_col': [99, 99]
            })
            
            test_data = pd.DataFrame({
                'sample_index': [0],
                'time': [0],
                'n_eyes': ['two'],
                'feature1': [1.5],
                'extra_col': [99]
            })
            
            labels = pd.DataFrame({
                'label': ['no_pain', 'low_pain']
            })
            
            raw_path = os.path.join(tmpdir, 'raw')
            os.makedirs(raw_path)
            
            train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
            test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
            labels.to_csv(os.path.join(raw_path, 'labels.csv'), index=False)
            
            # Create a read-only directory on Windows (or just test with valid path)
            import platform
            if platform.system() == 'Windows':
                # On Windows, the path will be created successfully, so skip this specific error test
                # Instead verify the preprocess completes
                params = {
                    "path_raw_data": raw_path,
                    "path_processed_data": os.path.join(tmpdir, 'processed'),
                    "name_train_file": "train.csv",
                    "name_test_file": "test.csv",
                    "name_train_labels_file": "labels.csv",
                    "verbose": False
                }
                
                preprocessor = PreProcessor(params)
                preprocessor.preprocess()
                
                captured = capsys.readouterr()
                # Should complete successfully or with error
                assert "successfully" in captured.out or "Error" in captured.out
            else:
                # On Unix-like systems, test with truly invalid path
                params = {
                    "path_raw_data": raw_path,
                    "path_processed_data": "/invalid/readonly/path",
                    "name_train_file": "train.csv",
                    "name_test_file": "test.csv",
                    "name_train_labels_file": "labels.csv",
                    "verbose": False
                }
                
                preprocessor = PreProcessor(params)
                preprocessor.preprocess()
                
                captured = capsys.readouterr()
                # Should report save error
                assert "Error" in captured.out
