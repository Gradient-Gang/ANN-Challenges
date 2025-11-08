import pytest
import optuna
import lightning as L
from src.GradientGang.Optimizer.OptunaOptimizer import OptunaOptimizer

# Mock classes
class MockLightningModule(L.LightningModule):
    def __init__(self, validation_result=0.5):
        super().__init__()
        self.validation_result = validation_result

    def validation_step(self):
        return self.validation_result

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
            "seq": ["option1", "option2", "option3"]
        },
        "float_param": {
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "step": 0.1,
            "log": False
        },
        "int_param": {
            "type": "int",
            "low": 1,
            "high": 10,
            "step": 1,
            "log": False
        }
    }

# Test getParams
def test_getParams_categorical(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"categorical_param": sample_params["categorical_param"]}, mock_trial)
    assert "categorical_param" in params
    assert params["categorical_param"] in sample_params["categorical_param"]["seq"]

def test_getParams_float(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"float_param": sample_params["float_param"]}, mock_trial)
    assert "float_param" in params
    assert isinstance(params["float_param"], float)
    assert sample_params["float_param"]["low"] <= params["float_param"] <= sample_params["float_param"]["high"]

def test_getParams_int(optimizer, mock_trial, sample_params):
    params = optimizer.getParams({"int_param": sample_params["int_param"]}, mock_trial)
    assert "int_param" in params
    assert isinstance(params["int_param"], int)
    assert sample_params["int_param"]["low"] <= params["int_param"] <= sample_params["int_param"]["high"]

def test_getParams_value_type(optimizer, mock_trial):
    value_params = {
        "constant_param": {
            "type": "value",
            "value": 42
        },
        "constant_str": {
            "type": "value",
            "value": "test_string"
        },
        "constant_float": {
            "type": "value",
            "value": 3.14
        }
    }
    params = optimizer.getParams(value_params, mock_trial)
    assert params["constant_param"] == 42
    assert params["constant_str"] == "test_string"
    assert params["constant_float"] == 3.14

def test_getParams_invalid_type(optimizer, mock_trial):
    invalid_params = {
        "invalid_param": {
            "type": "invalid",
            "value": 42
        }
    }
    with pytest.raises(TypeError, match="Pipeline.getParams: Invalid type for parameter invalid_param"):
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

    study = optuna.create_study()
    study.optimize(lambda t: 0.5, n_trials=1)  # Pre-optimize to avoid hanging
    
    result = optimizer.optimize(mock_architecture_builder, {"test_param": {"type": "float", "low": 0, "high": 1, "log": False}})
    assert isinstance(result, optuna.study.Study)
    assert len(result.trials) > 0
    assert result.best_value == 0.5  # Since our mock always returns 0.5

# Test objective
def test_objective(optimizer, mock_trial):
    def mock_architecture_builder(params):
        return MockLightningModule(validation_result=0.75)

    optimizer.build_architecture = mock_architecture_builder
    optimizer.params = {"test_param": {"type": "float", "low": 0, "high": 1, "step": 0.1, "log": False}}

    result = optimizer.objective(mock_trial)
    assert result == 0.75

def test_objective_with_different_validation_results(optimizer, mock_trial):
    test_values = [0.0, 0.5, 1.0]
    
    for val in test_values:
        def mock_architecture_builder(params):
            return MockLightningModule(validation_result=val)

        optimizer.build_architecture = mock_architecture_builder
        optimizer.params = {"test_param": {"type": "float", "low": 0, "high": 1, "step": 0.1, "log": False}}

        result = optimizer.objective(mock_trial)
        assert result == val
