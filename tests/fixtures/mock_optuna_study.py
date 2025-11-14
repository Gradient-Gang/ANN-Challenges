"""Mock Optuna study and trial fixtures for testing."""

import pytest
from unittest.mock import Mock
import optuna


@pytest.fixture
def mock_optuna_trial():
    """Create mock Optuna trial with best hyperparameters."""
    mock_trial = Mock()
    mock_trial.number = 0
    mock_trial.value = 0.90
    mock_trial.params = {
        "MacroArchitecture": "Direct",
        "architectureType": "Recurrent",
        "rnnType": "LSTM",
        "hiddenDim": 128,
        "numLayers": 2,
        "bidirectional": False,
        "recurrentDropout": 0.2,
        "encoderActivation": "ReLU",
        "globalEmbeddingDim": 32,
        "globalNumLayers": 2,
        "globalDropout": 0.1,
        "globalActivation": "ReLU",
        "numFFLayers": 2,
        "ffHiddenDim": 64,
        "ffDropout": 0.3,
        "ffActivation": "ReLU",
        "use_windowing": False,
        "LearningRate": 0.001,
        "RegularizationWeight": 0.01,
        "SchedulerType": "ReduceLROnPlateau",
        "SchedulerPatience": 5,
        "SchedulerFactor": 0.5,
        "SchedulerMinLR": 1e-6,
        "EarlyStoppingPatience": 10,
    }
    mock_trial.user_attrs = {
        "fold_scores": "0.90,0.88,0.89,0.91,0.87",
        "mean_f1": 0.89,
        "std_f1": 0.015,
    }
    mock_trial.state = optuna.trial.TrialState.COMPLETE

    return mock_trial


@pytest.fixture
def mock_optuna_study(mock_optuna_trial):
    """Create mock Optuna study with completed trial."""
    mock_study = Mock()
    mock_study.best_trial = mock_optuna_trial
    mock_study.trials = [mock_optuna_trial]
    mock_study.study_name = "test_study"

    return mock_study
