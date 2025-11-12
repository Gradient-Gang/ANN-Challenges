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
            # Create sample training data with sample_index
            train_data = pd.DataFrame({
                'sample_index': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                'time': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'feature2': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                'feature3': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
            })
            train_data.to_csv(os.path.join(tmpdir, 'train.csv'), index=False)

            # Create sample training labels
            train_labels = pd.DataFrame({
                'sample_index': list(range(5)),
                'label': ['no_pain', 'low_pain', 'no_pain', 'low_pain', 'no_pain']
            })
            train_labels.to_csv(os.path.join(tmpdir, 'train_labels.csv'), index=False)

            # Create sample test data with sample_index
            test_data = pd.DataFrame({
                'sample_index': [0, 0, 1, 1, 2, 2],
                'time': [0, 1, 0, 1, 0, 1],
                'feature1': [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                'feature2': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                'feature3': [0.0, -1.0, -2.0, -3.0, -4.0, -5.0]
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

    def test_dataloader_setup(self, sample_data_dir):
        """Test DataModule setup creates datasets."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        dm.setup()
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_dataloader_train_dataloader(self, sample_data_dir):
        """Test train_dataloader returns DataLoader."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        dm.setup()
        
        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 2

    def test_dataloader_val_dataloader(self, sample_data_dir):
        """Test val_dataloader returns DataLoader."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        dm.setup()
        
        val_loader = dm.val_dataloader()
        assert val_loader is not None

    def test_dataloader_test_dataloader(self, sample_data_dir):
        """Test test_dataloader returns DataLoader."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        dm.setup()
        
        test_loader = dm.test_dataloader()
        assert test_loader is not None

    def test_dataloader_predict_dataloader(self, sample_data_dir):
        """Test predict_dataloader returns test DataLoader."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        dm.setup()
        
        predict_loader = dm.predict_dataloader()
        assert predict_loader is not None
        # Should be same as test_dataloader
        test_loader = dm.test_dataloader()
        assert predict_loader.dataset == test_loader.dataset

    def test_dataloader_without_setup_raises_error(self, sample_data_dir):
        """Test that accessing dataloaders without setup raises AttributeError or RuntimeError."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        dm = DataModule(params)
        
        # Should raise AttributeError or RuntimeError when datasets not initialized
        with pytest.raises((RuntimeError, AttributeError)):
            dm.train_dataloader()
        
        with pytest.raises((RuntimeError, AttributeError)):
            dm.val_dataloader()
        
        with pytest.raises((RuntimeError, AttributeError)):
            dm.test_dataloader()

    def test_dataloader_cuda_pin_memory(self, sample_data_dir, monkeypatch):
        """Test that pin_memory is set based on CUDA availability."""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "val_split": 0.2
        }
        
        # Mock CUDA availability
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        
        dm = DataModule(params)
        dm.setup()
        
        train_loader = dm.train_dataloader()
        assert train_loader.pin_memory is True


