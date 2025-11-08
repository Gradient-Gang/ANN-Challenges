import pytest
import optuna
import lightning as L
from GradientGang.Pipeline.Optimizer.OptunaOptimizer import OptunaOptimizer

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
            "params": {
                "name": "categorical_param",
                "choices": ["option1", "option2", "option3"]
            }
        },
        "float_param": {
            "type": "float",
            "params": {
                "name": "float_param",
                "low": 0.0,
                "high": 1.0,
                "step": 0.1,
                "log": False
            }
        },
        "int_param": {
            "type": "int",
            "params": {
                "name": "int_param",
                "low": 1,
                "high": 10,
                "step": 1,
                "log": False
            }
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

    test_params = {
        "test_param": {
            "type": "float",
            "params": {
                "name": "test_param",
                "low": 0,
                "high": 1,
                "log": False
            }
        }
    }
    
    result = optimizer.optimize(mock_architecture_builder, test_params, n_trials=1)  # Specify n_trials to avoid infinite loop
    assert isinstance(result, optuna.study.Study)
    assert len(result.trials) == 1
    assert result.best_value == 0.5  # Since our mock always returns 0.5

# Test objective
def test_objective(optimizer, mock_trial):
    def mock_architecture_builder(params):
        return MockLightningModule(validation_result=0.75)

    optimizer.build_architecture = mock_architecture_builder
    optimizer.params = {
        "test_param": {
            "type": "float",
            "params": {
                "name": "test_param",
                "low": 0,
                "high": 1,
                "step": 0.1,
                "log": False
            }
        }
    }

    result = optimizer.objective(mock_trial)
    assert result == 0.75

def test_objective_with_different_validation_results(optimizer, mock_trial):
    test_values = [0.0, 0.5, 1.0]
    
    for val in test_values:
        def mock_architecture_builder(params):
            return MockLightningModule(validation_result=val)

        optimizer.build_architecture = mock_architecture_builder
        optimizer.params = {
            "test_param": {
                "type": "float",
                "params": {
                    "name": "test_param",
                    "low": 0,
                    "high": 1,
                    "step": 0.1,
                    "log": False
                }
            }
        }

        result = optimizer.objective(mock_trial)
        assert result == val
