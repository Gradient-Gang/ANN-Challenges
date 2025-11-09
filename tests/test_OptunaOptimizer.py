import pytest
import optuna
import torch
import lightning as L
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from GradientGang.Pipeline.Optimizer.OptunaOptimizer import OptunaOptimizer

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

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.0, requires_grad=True)  # Mock training step with proper tensor
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_f1", self.validation_result)  # Log during validation step
        return {"val_f1": self.validation_result}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # Mock optimizer with actual optimizer instance
        return optimizer

# Fixtures
@pytest.fixture
def optimizer():
    return OptunaOptimizer()

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

# Test getParams
def test_getParams_categorical(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"categorical_param": sample_params["categorical_param"]}, mock_trial)
    assert "categorical_param" in params
    assert params["categorical_param"] in sample_params["categorical_param"]["params"]["choices"]

def test_getParams_float(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"float_param": sample_params["float_param"]}, mock_trial)
    assert "float_param" in params
    assert isinstance(params["float_param"], float)
    assert sample_params["float_param"]["params"]["low"] <= params["float_param"] <= sample_params["float_param"]["params"]["high"]

def test_getParams_int(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"int_param": sample_params["int_param"]}, mock_trial)
    assert "int_param" in params
    assert isinstance(params["int_param"], int)
    assert sample_params["int_param"]["params"]["low"] <= params["int_param"] <= sample_params["int_param"]["params"]["high"]

def test_getParams_missing_params(optimizer, mock_trial):
    invalid_params = {
        "param": {
            "type": "float"  # Missing params dictionary
        }
    }
    with pytest.raises(KeyError, match="Required parameter 'params' not found in provided parameters"):
        optimizer.getParams(invalid_params, mock_trial)

def test_getParams_invalid_type(optimizer, mock_trial):
    invalid_params = {
        "invalid_param": {
            "type": "invalid",
            "params": {}
        }
    }
    with pytest.raises(KeyError, match="ParameterInterpreter: 'invalid' not found in interpretation dictionary"):
        optimizer.getParams(invalid_params, mock_trial)

def test_getParams_all_types(optimizer, mock_trial, sample_params):
    params = optimizer.getParams(sample_params, mock_trial)
    assert len(params) == 3
    assert "categorical_param" in params
    assert "float_param" in params
    assert "int_param" in params

# Test optimize
def test_optimize(optimizer):
    def mock_architecture_builder(params):
        return MockLightningModule(validation_result=0.5)

    # Create DataLoaders from MockDataset
    mock_train_data = DataLoader(MockDataset(), batch_size=1)
    mock_val_data = DataLoader(MockDataset(), batch_size=1)

    test_params = {
        "test_param": {
            "type": "float",
            "params": {
                "name": "test_param",
                "low": 0,
                "high": 1,
                "log": False
            },
            "value": 0.5
        }
    }
    
    result = optimizer.optimize(mock_architecture_builder, test_params, train_data=mock_train_data, 
                             val_data=mock_val_data, n_trials=1)  # Specify n_trials to avoid infinite loop
    assert isinstance(result, optuna.study.Study)
    assert len(result.trials) == 1
    assert result.best_value == 0.5  # Since our mock always returns 0.5

# Test objective
def test_objective(optimizer, mock_trial):
    def mock_architecture_builder(params):
        return MockLightningModule(validation_result=0.75)

    # Create DataLoaders from MockDataset
    mock_train_data = DataLoader(MockDataset(), batch_size=1)
    mock_val_data = DataLoader(MockDataset(), batch_size=1)

    optimizer.build_architecture = mock_architecture_builder
    optimizer.train_data = mock_train_data
    optimizer.val_data = mock_val_data
    optimizer.params = {
        "test_param": {
            "type": "float",
            "params": {
                "name": "test_param",
                "low": 0,
                "high": 1,
                "log": False
            },
            "value": 0.5
        }
    }

    result = optimizer.objective(mock_trial)
    assert result == 0.75

def test_objective_with_different_validation_results(optimizer, mock_trial):
    test_values = [0.0, 0.5, 1.0]
    
        # Create DataLoaders from MockDataset
    mock_train_data = DataLoader(MockDataset(), batch_size=1)
    mock_val_data = DataLoader(MockDataset(), batch_size=1)
    
    for val in test_values:
        def mock_architecture_builder(params):
            return MockLightningModule(validation_result=val)

        optimizer.build_architecture = mock_architecture_builder
        optimizer.train_data = mock_train_data
        optimizer.val_data = mock_val_data
        optimizer.params = {
            "test_param": {
                "type": "float",
                "params": {
                    "name": "test_param",
                    "low": 0,
                    "high": 1,
                    "log": False
                },
                "value": 0.5
            }
        }

        result = optimizer.objective(mock_trial)
        assert result == val
