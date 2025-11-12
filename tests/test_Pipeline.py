
import pytest
import lightning as L
from GradientGang.Pipeline.Pipeline import Pipeline
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Architectures.Direct import Direct
import yaml
import tempfile
import os

# Mock classes for testing
class MockDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

# Fixtures
@pytest.fixture
def dict_loader_config():
    return {
        "data_dir": "dummy_data",
        "train_file_name": "dummy_train.csv",
        "train_file_name_labels": "dummy_labels.csv",
        "test_file_name": "dummy_test.csv",
        "batch_size": 32,
        "num_workers": 0,
        "val_split": 0.1
    }

@pytest.fixture
def valid_arch_params():
    return {
        "arch_type": "direct",
        "OutputDim": 10,
        "RegularizationWeight": 0.1,
        "EncoderParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 64,
                    "kernel_size": 3
                }
            }]
        },
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 1,
                    "out_features": 1
                }
            }]
        },
        "FeedForwardParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 784,
                    "out_features": 10
                }
            }]
        },
        "LearningRate": 0.001,
        "Patience": 3
    }

# Test constructor validation
def test_constructor_with_dict_config(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    assert pipeline.data_loader is not None
    assert hasattr(pipeline, 'build_architecture')
    assert hasattr(pipeline, 'fit_and_validate')

def test_constructor_creates_datamodule(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    # Verify that data_loader is created
    assert pipeline.data_loader is not None
    from GradientGang.Pipeline.DataLoader.DataLoader import DataModule
    assert isinstance(pipeline.data_loader, DataModule)

# Test build_architecture
def test_build_architecture_autoencoder_split(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    params = {
        "arch_type": "autoencoder_split",
        "OutputDim": 10,
        "RegularizationWeight": 0.1,
        "ReconstructionLossWeight": 0.5,
        "EncoderParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 64,
                    "kernel_size": 3
                }
            }]
        },
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 1,
                    "out_features": 1
                }
            }]
        },
        "DecoderParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "ConvTranspose2d",
                "params": {
                    "in_channels": 64,
                    "out_channels": 1,
                    "kernel_size": 3
                }
            }]
        },
        "GlobalFFDecoderParams": {
            "activation_function": "ReLU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 1,
                    "out_features": 1
                }
            }]
        },
        "FeedForwardParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 784,
                    "out_features": 10
                }
            }]
        },
        "LearningRate": 0.001,
        "Patience": 3
    }
    arch = pipeline.build_architecture(params)
    assert isinstance(arch, LightningAutoencoder)

def test_build_architecture_autoencoder_joint(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    params = {
        "arch_type": "autoencoder_joint",
        "OutputDim": 10,
        "RegularizationWeight": 0.1,
        "ReconstructionLossWeight": 0.5,
        "EncoderParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Conv2d",
                "params": {
                    "in_channels": 1,
                    "out_channels": 64,
                    "kernel_size": 3
                }
            }]
        },
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 1,
                    "out_features": 1
                }
            }]
        },
        "DecoderParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "ConvTranspose2d",
                "params": {
                    "in_channels": 64,
                    "out_channels": 1,
                    "kernel_size": 3
                }
            }]
        },
        "GlobalFFDecoderParams": {
            "activation_function": "ReLU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 1,
                    "out_features": 1
                }
            }]
        },
        "FeedForwardParams": {
            "activation_function": "GELU",
            "layer_type": [{
                "name": "Linear",
                "params": {
                    "in_features": 784,
                    "out_features": 10
                }
            }]
        },
        "LearningRate": 0.001,
        "Patience": 3
    }
    arch = pipeline.build_architecture(params)
    assert isinstance(arch, LightningAutoencoder)

def test_build_architecture_direct(dict_loader_config, valid_arch_params):
    pipeline = Pipeline(dict_loader_config)
    arch = pipeline.build_architecture(valid_arch_params)
    assert isinstance(arch, Direct)

def test_build_architecture_invalid_type(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    params = {"arch_type": "invalid_type"}
    with pytest.raises(KeyError):
        pipeline.build_architecture(params)

# Test that Pipeline has correct methods
def test_pipeline_has_fit_and_validate(dict_loader_config):
    pipeline = Pipeline(dict_loader_config)
    assert hasattr(pipeline, 'fit_and_validate')
    assert callable(pipeline.fit_and_validate)

def test_build_architecture_different_output_dims(dict_loader_config, valid_arch_params):
    """Test building architectures with different output dimensions."""
    pipeline = Pipeline(dict_loader_config)
    
    for output_dim in [2, 5, 10]:
        params = valid_arch_params.copy()
        params["OutputDim"] = output_dim
        arch = pipeline.build_architecture(params)
        assert arch is not None

def test_build_architecture_missing_arch_type(dict_loader_config):
    """Test that missing arch_type raises KeyError."""
    pipeline = Pipeline(dict_loader_config)
    params = {"OutputDim": 10}  # Missing arch_type
    
    with pytest.raises(KeyError):
        pipeline.build_architecture(params)

def test_pipeline_multiple_builds(dict_loader_config, valid_arch_params):
    """Test building multiple architectures from same pipeline."""
    pipeline = Pipeline(dict_loader_config)
    
    arch1 = pipeline.build_architecture(valid_arch_params)
    arch2 = pipeline.build_architecture(valid_arch_params)
    
    # They should be different instances
    assert arch1 is not arch2


# Test fit_and_validate method
def test_fit_and_validate_has_correct_signature(dict_loader_config):
    """Test that fit_and_validate method exists with correct signature."""
    pipeline = Pipeline(dict_loader_config)
    import inspect
    sig = inspect.signature(pipeline.fit_and_validate)
    params = list(sig.parameters.keys())
    assert 'dict_arch' in params
    assert 'dict_data' in params


def test_fit_and_validate_builds_architecture(dict_loader_config, valid_arch_params, monkeypatch):
    """Test that fit_and_validate builds the architecture."""
    pipeline = Pipeline(dict_loader_config)
    
    # Track if build_architecture was called
    build_called = []
    original_build = pipeline.build_architecture
    
    def mock_build(params):
        build_called.append(params)
        return original_build(params)
    
    monkeypatch.setattr(pipeline, 'build_architecture', mock_build)
    
    # Mock trainer to avoid actual training
    class MockTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr(L, 'Trainer', MockTrainer)
    
    dict_data = {
        "test_split": 0.2,
        "is_3d": True,
        "seed": 42
    }
    
    try:
        pipeline.fit_and_validate(valid_arch_params, dict_data)
    except Exception:
        pass  # We expect some errors due to mocking
    
    # Verify build_architecture was called
    assert len(build_called) > 0
    assert build_called[0] == valid_arch_params


def test_fit_and_validate_configures_loader(dict_loader_config, valid_arch_params, monkeypatch):
    """Test that fit_and_validate configures the data loader."""
    pipeline = Pipeline(dict_loader_config)
    
    # Track if setup was called
    setup_called = []
    original_setup = pipeline.data_loader.setup
    
    def mock_setup(**kwargs):
        setup_called.append(kwargs)
        return original_setup(**kwargs)
    
    monkeypatch.setattr(pipeline.data_loader, 'setup', mock_setup)
    
    # Mock trainer
    class MockTrainer:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
    
    monkeypatch.setattr(L, 'Trainer', MockTrainer)
    
    dict_data = {
        "test_split": 0.2,
        "is_3d": True,
        "seed": 42
    }
    
    try:
        pipeline.fit_and_validate(valid_arch_params, dict_data)
    except Exception:
        pass
    
    # Verify setup was called with correct params
    assert len(setup_called) > 0
    assert setup_called[0] == dict_data

