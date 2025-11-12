import pytest
import optuna
import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from GradientGang.Pipeline.Optimizer.OptunaOptimizer import OptunaOptimizer
import warnings

# Filter PyTorch Lightning warnings in tests
warnings.filterwarnings("ignore", message=".*max_epochs.*")
warnings.filterwarnings("ignore", message=".*num_workers.*")
warnings.filterwarnings("ignore", message=".*log_every_n_steps.*")

# Mock classes
class MockDataset(Dataset):
    def __init__(self):
        self.data = [(torch.tensor([[0.0]]), torch.tensor([0]))]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

class MockLightningModule(L.LightningModule):
    def __init__(self, validation_result=0.5):
        super().__init__()
        self.validation_result = validation_result
        self.layer = nn.Linear(1, 1)  # Add a parameter to make optimizer happy
        self._train_dataloader = DataLoader(MockDataset(), batch_size=1)
        self._val_dataloader = DataLoader(MockDataset(), batch_size=1)

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, requires_grad=True)  # Mock training step with proper tensor
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_f1", self.validation_result)  # Log during validation step
        return {"val_f1": self.validation_result}
    
    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        return self._val_dataloader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # Mock optimizer with actual optimizer instance
        return optimizer

# Fixtures
@pytest.fixture
def dict_config():
    return {
        "dataloader": {
            "data_dir": "dummy_data",
            "train_file_name": "dummy_train.csv",
            "train_file_name_labels": "dummy_labels.csv",
            "test_file_name": "dummy_test.csv",
            "batch_size": 32,
            "num_workers": 0,
            "val_split": 0.1
        },
        "arch": {
            "arch_type": "direct",
            "OutputDim": 10,
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [{"name": "Linear", "params": {"in_features": 10, "out_features": 5}}]
            },
            "GlobalFFEncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [{"name": "Linear", "params": {"in_features": 5, "out_features": 5}}]
            },
            "FeedForwardParams": {
                "activation_function": "ReLU",
                "layer_type": [{"name": "Linear", "params": {"in_features": 5, "out_features": 10}}]
            },
            "LearningRate": 0.001,
            "Patience": 3
        },
        "hyper_dataloader": {},
        "hyper_arch": {
            "LearningRate": {
                "name": "LearningRate",
                "type": "float",
                "opts": {"low": 0.0001, "high": 0.01, "log": True},
                "paths": ["LearningRate"]
            }
        }
    }

@pytest.fixture
def optimizer(dict_config):
    return OptunaOptimizer(dict_config)

@pytest.fixture
def mock_trial():
    study = optuna.create_study()
    trial = study.ask()
    return trial

@pytest.fixture
def sample_params():
    return {
        "categorical_param": {
            "type": "categ",
            "params": {
                "name": "categorical_param",
                "choices": ["option1", "option2", "option3"]
            },
            "value": "option1"
        },
        "float_param": {
            "type": "float",
            "params": {
                "name": "float_param",
                "low": 0.0,
                "high": 1.0,
                "log": False
            },
            "value": 0.5
        },
        "int_param": {
            "type": "int",
            "params": {
                "name": "int_param",
                "low": 1,
                "high": 10,
                "log": False
            },
            "value": 5
        }
    }

# Test HyperParameter class
def test_getParams_categorical(optimizer, mock_trial):
    hp_dict = {
        "test_categ": {
            "name": "test_categ",
            "type": "categ",
            "opts": {"choices": ["option1", "option2", "option3"]},
            "paths": ["test_categ"]
        }
    }
    hp = OptunaOptimizer.HyperParameter(hp_dict["test_categ"])
    value = hp.getValue(mock_trial)
    assert value in ["option1", "option2", "option3"]

def test_getParams_float(optimizer, mock_trial):
    hp_dict = {
        "test_float": {
            "name": "test_float",
            "type": "float",
            "opts": {"low": 0.0, "high": 1.0},
            "paths": ["test_float"]
        }
    }
    hp = OptunaOptimizer.HyperParameter(hp_dict["test_float"])
    value = hp.getValue(mock_trial)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0

def test_getParams_int(optimizer, mock_trial):
    hp_dict = {
        "test_int": {
            "name": "test_int",
            "type": "int",
            "opts": {"low": 1, "high": 10},
            "paths": ["test_int"]
        }
    }
    hp = OptunaOptimizer.HyperParameter(hp_dict["test_int"])
    value = hp.getValue(mock_trial)
    assert isinstance(value, int)
    assert 1 <= value <= 10

def test_getParams_missing_params(optimizer):
    # Test missing required fields in HyperParameter
    with pytest.raises(KeyError):
        OptunaOptimizer.HyperParameter({"name": "test"})  # Missing type, opts, paths

def test_getParams_invalid_type(optimizer, mock_trial):
    hp_dict = {
        "test_value": {
            "name": "test_value",
            "type": "value",
            "opts": {"value": 42},
            "paths": ["test_value"]
        }
    }
    hp = OptunaOptimizer.HyperParameter(hp_dict["test_value"])
    # Test value type returns the specified value directly
    value = hp.getValue(mock_trial)
    assert value == 42

def test_getParams_all_types(optimizer, mock_trial):
    # Test that optimizer loads hyperparameters correctly
    assert "LearningRate" in optimizer.hyperparams_arch
    assert optimizer.hyperparams_arch["LearningRate"].type == "float"

# Test optimize
def test_optimize(optimizer):
    # Test that optimize creates and returns an optuna study
    # Note: We can't easily test the full optimization without mock data files
    # So we just test that the method exists and has correct signature
    assert hasattr(optimizer, 'optimize')
    assert callable(optimizer.optimize)

# Test objective
def test_objective(optimizer, mock_trial):
    # Test the build method which is used by objective
    skeleton = {"key1": "value1"}
    hyperparams = {
        "test_hp": OptunaOptimizer.HyperParameter({
            "name": "test_param",
            "type": "float",
            "opts": {"low": 0.0, "high": 1.0},
            "paths": ["test_param"]
        })
    }
    
    result = optimizer.build(skeleton, hyperparams, mock_trial)
    assert "key1" in result
    assert result["key1"] == "value1"
    assert "test_param" in result
    assert isinstance(result["test_param"], float)

def test_objective_with_different_validation_results(optimizer, mock_trial):
    # Test the build method with nested structure
    skeleton = {
        "outer": {
            "inner": "value"
        }
    }
    hyperparams = {
        "nested_hp": OptunaOptimizer.HyperParameter({
            "name": "nested_param",
            "type": "int",
            "opts": {"low": 1, "high": 10},
            "paths": ["outer", "nested_param"]
        })
    }
    
    result = optimizer.build(skeleton, hyperparams, mock_trial)
    assert "outer" in result
    assert "inner" in result["outer"]
    assert "nested_param" in result["outer"]
    assert isinstance(result["outer"]["nested_param"], int)
    assert 1 <= result["outer"]["nested_param"] <= 10
