import pytest
import torch
import pandas as pd
import os
import tempfile
from src.GradientGang.Pipeline.DataLoader import DataModule


class TestDataModule:
    """Test suite for DataModule class"""

    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample training data
            train_data = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'feature2': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                'feature3': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            })
            train_data.to_csv(os.path.join(tmpdir, 'train.csv'), index=False)

            # Create sample training labels
            train_labels = pd.DataFrame({
                'sample_index': list(range(10)),
                'label': ['no_pain', 'low_pain', 'no_pain', 'low_pain', 'no_pain',
                         'low_pain', 'no_pain', 'low_pain', 'no_pain', 'low_pain']
            })
            train_labels.to_csv(os.path.join(tmpdir, 'train_labels.csv'), index=False)

            # Create sample test data
            test_data = pd.DataFrame({
                'feature1': [11.0, 12.0, 13.0],
                'feature2': [10.5, 11.5, 12.5],
                'feature3': [0.0, -1.0, -2.0]
            })
            test_data.to_csv(os.path.join(tmpdir, 'test.csv'), index=False)

            yield tmpdir

    def test_dataloader_initialization(self, sample_data_dir):
        """Test DataModule initialization with valid parameters"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.1
        }
        
        dm = DataModule(params)
        assert dm.data_dir == sample_data_dir
        assert dm.batch_size == 2
        assert dm.num_workers == 0
        assert dm.val_split == 0.1  # default value

    def test_dataloader_missing_required_params(self):
        """Test that missing required parameters raises KeyError"""
        params = {
            "data_dir": "/some/path",
            "batch_size": 32
        }
        
        with pytest.raises(KeyError):
            DataModule(params)

