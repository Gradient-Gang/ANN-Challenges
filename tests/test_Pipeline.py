import pytest
import lightning as L
from GradientGang.Pipeline.Pipeline import Pipeline
from GradientGang.Pipeline.Optimizer.Optimizer import Optimizer
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Architectures.Direct import Direct
import yaml
import tempfile
import os

# Mock classes for testing
class MockDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

class MockOptimizer(Optimizer):
    def optimize(self, architecture_builder, params):
        self.architecture_builder = architecture_builder
        self.params = params
        return "mock_study"

# Fixtures
@pytest.fixture
def mock_dataset():
    return MockDataModule()

@pytest.fixture
def mock_test_dataset():
    return MockDataModule()

@pytest.fixture
def mock_optimizer():
    return MockOptimizer()

@pytest.fixture
def valid_config():
    return {
        "arch_type": {
            "type": "categ",
            "seq": ["autoencoder_split", "autoencoder_joint", "direct"]
        }
    }

@pytest.fixture
def config_file(valid_config):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.dump(valid_config, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

# Test constructor validation
def test_constructor_with_dict_config(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    assert pipeline.config == valid_config
    assert pipeline.dataset == mock_dataset
    assert pipeline.test == mock_test_dataset
    assert pipeline.optimizer == mock_optimizer

def test_constructor_with_path_config(mock_dataset, mock_test_dataset, mock_optimizer, config_file):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, path_config=config_file)
    assert isinstance(pipeline.config, dict)
    assert pipeline.dataset == mock_dataset
    assert pipeline.test == mock_test_dataset
    assert pipeline.optimizer == mock_optimizer

def test_constructor_invalid_both_configs(mock_dataset, mock_test_dataset, mock_optimizer, valid_config, config_file):
    with pytest.raises(ValueError, match="dict_config or path_config have to be assigned"):
        Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config, path_config=config_file)

def test_constructor_no_config(mock_dataset, mock_test_dataset, mock_optimizer):
    with pytest.raises(ValueError, match="dict_config or path_config have to be assigned"):
        Pipeline(mock_dataset, mock_test_dataset, mock_optimizer)

def test_constructor_invalid_dict_config(mock_dataset, mock_test_dataset, mock_optimizer):
    with pytest.raises(ValueError, match="dict_config must be a dictionary"):
        Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config="not_a_dict")

def test_constructor_invalid_path_config(mock_dataset, mock_test_dataset, mock_optimizer):
    with pytest.raises(ValueError, match="path_config must be a string"):
        Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, path_config=123)

def test_constructor_invalid_dataset_type(mock_optimizer, valid_config):
    with pytest.raises(ValueError, match="dataset and test must be L.LightningDataModule"):
        Pipeline("not_a_dataset", MockDataModule(), mock_optimizer, dict_config=valid_config)

def test_constructor_invalid_test_type(mock_dataset, mock_optimizer, valid_config):
    with pytest.raises(ValueError, match="dataset and test must be L.LightningDataModule"):
        Pipeline(mock_dataset, "not_a_test_dataset", mock_optimizer, dict_config=valid_config)

# Test build_architecture
def test_build_architecture_autoencoder_split(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    params = {
        "arch_type": "autoencoder_split",
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

def test_build_architecture_autoencoder_joint(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    params = {
        "arch_type": "autoencoder_joint",
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

def test_build_architecture_direct(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    params = {
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
    arch = pipeline.build_architecture(params)
    assert isinstance(arch, Direct)

def test_build_architecture_invalid_type(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    params = {"arch_type": "invalid_type"}
    with pytest.raises(KeyError):
        pipeline.build_architecture(params)

# Test optimize
def test_optimize(mock_dataset, mock_test_dataset, mock_optimizer, valid_config):
    pipeline = Pipeline(mock_dataset, mock_test_dataset, mock_optimizer, dict_config=valid_config)
    result = pipeline.optimize()
    assert result == "mock_study"  # Verify the mock optimizer was called and returned expected value
    assert mock_optimizer.architecture_builder == pipeline.build_architecture
    assert mock_optimizer.params == valid_config
