import pytest
import torch
import pandas as pd
import os
import tempfile
from src.GradientGang.Pipeline.DataLoader import DataModule, DataLoaderInterpreter


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
            "num_workers": 0
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

    def test_load_from_csv(self, sample_data_dir):
        """Test loading data from CSV files"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0
        }
        
        dm = DataModule(params)
        
        # Test loading with labels
        train_path = os.path.join(sample_data_dir, "train.csv")
        train_labels_path = os.path.join(sample_data_dir, "train_labels.csv")
        features, labels = dm.load_from_csv(train_path, train_labels_path)
        
        assert features.shape == (10, 3)  # 10 samples, 3 features
        assert labels.shape == (10,)
        assert features.dtype == torch.float32
        assert labels.dtype == torch.long

    def test_label_mapping(self, sample_data_dir):
        """Test that string labels are correctly mapped to integers"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "label_mapping": {
                "no_pain": 0,
                "low_pain": 1
            }
        }
        
        dm = DataModule(params)
        train_path = os.path.join(sample_data_dir, "train.csv")
        train_labels_path = os.path.join(sample_data_dir, "train_labels.csv")
        features, labels = dm.load_from_csv(train_path, train_labels_path)
        
        # Check that labels are correctly mapped
        assert torch.all((labels == 0) | (labels == 1))

    def test_setup_fit(self, sample_data_dir):
        """Test setup method for training stage"""
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
        dm.setup(stage='fit')
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 8  # 80% of 10
        assert len(dm.val_dataset) == 2   # 20% of 10

    def test_setup_test(self, sample_data_dir):
        """Test setup method for testing stage"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0
        }
        
        dm = DataModule(params)
        dm.setup(stage='test')
        
        assert dm.test_dataset is not None
        assert len(dm.test_dataset) == 3  # 3 test samples

    def test_dataloaders(self, sample_data_dir):
        """Test that dataloaders are created correctly"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0
        }
        
        dm = DataModule(params)
        dm.setup()
        
        # Test train dataloader
        train_loader = dm.train_dataloader()
        assert train_loader.batch_size == 2
        
        # Test validation dataloader
        val_loader = dm.val_dataloader()
        assert val_loader.batch_size == 2
        
        # Test test dataloader
        test_loader = dm.test_dataloader()
        assert test_loader.batch_size == 2
        
        # Test that we can iterate through a batch
        batch = next(iter(train_loader))
        features, labels = batch
        assert features.shape[0] <= 2  # batch size
        assert features.shape[1] == 3  # number of features
        assert labels.shape[0] <= 2

    def test_dataloader_without_setup_raises_error(self, sample_data_dir):
        """Test that calling dataloader without setup raises RuntimeError"""
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0
        }
        
        dm = DataModule(params)
        
        with pytest.raises(RuntimeError):
            dm.train_dataloader()

    def test_custom_label_mapping(self, sample_data_dir):
        """Test using custom label mapping"""
        custom_mapping = {
            "no_pain": 10,
            "low_pain": 20,
            "medium_pain": 30,
            "high_pain": 40
        }
        
        params = {
            "data_dir": sample_data_dir,
            "train_file_name": "train.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test.csv",
            "batch_size": 2,
            "num_workers": 0,
            "label_mapping": custom_mapping
        }
        
        dm = DataModule(params)
        train_path = os.path.join(sample_data_dir, "train.csv")
        train_labels_path = os.path.join(sample_data_dir, "train_labels.csv")
        features, labels = dm.load_from_csv(train_path, train_labels_path)
        
        # Check that labels use custom mapping
        assert torch.all((labels == 10) | (labels == 20))
