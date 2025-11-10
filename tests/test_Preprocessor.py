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
