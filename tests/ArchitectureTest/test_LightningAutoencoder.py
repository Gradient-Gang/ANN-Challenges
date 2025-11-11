import pytest
import torch
import pytorch_lightning as L
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


def get_basic_params():
    """Helper to get basic valid params."""
    return {
        "LearningRate": 0.001,
        "Patience": 5,
        "RegularizationWeight": 0.1,
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 1,
                        "out_features": 1
                    }
                }
            ]
        },
        "EncoderParams": {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Flatten",
                    "params": {}
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 128,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "DecoderParams": {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 784,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "FeedForwardParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 129,  # 128 from encoder + 1 from global features
                        "out_features": 64,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 10,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "OutputDim": 10,
        "ReconstructionLossWeight": 0.5,
        "GlobalFFDecoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 1,
                        "out_features": 1
                    }
                }
            ]
        }
    }


class TestLightningAutoencoderInitialization:
    """Test LightningAutoencoder initialization."""

    def test_basic_initialization(self):
        """Test basic autoencoder initialization."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        assert model is not None
        assert model.reconstruction_loss_weight == 0.5

    def test_invalid_reconstruction_weight_high(self):
        """Test that reconstruction weight > 1 raises error."""
        params = get_basic_params()
        params["ReconstructionLossWeight"] = 1.5
        with pytest.raises(AssertionError):
            LightningAutoencoder(params)

    def test_invalid_reconstruction_weight_negative(self):
        """Test that reconstruction weight < 0 raises error."""
        params = get_basic_params()
        params["ReconstructionLossWeight"] = -0.1
        with pytest.raises(AssertionError):
            LightningAutoencoder(params)

    def test_default_class_weights(self):
        """Test default class weights are all ones."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        assert torch.allclose(model.class_weights, torch.ones(10))


class TestLightningAutoencoderForward:
    """Test forward pass."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        model.eval()
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 784)
        global_features = torch.randn(batch_size, 1)
        
        predictions, (decoded, decoded_global) = model((time_series, global_features))
        
        assert predictions.shape == (batch_size, 10)
        assert decoded.shape[0] == batch_size
        assert decoded_global.shape == (batch_size, 1)


class TestLightningAutoencoderTraining:
    """Test training functionality."""

    def test_training_step_labeled(self):
        """Test training step with labeled data."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 784)
        global_features = torch.randn(batch_size, 1)
        labels = torch.randint(0, 10, (batch_size,))
        
        batch = ((time_series, global_features), labels)
        loss = model.training_step(batch, 0)
        
        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_training_step_unlabeled(self):
        """Test training step with unlabeled data."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 784)
        global_features = torch.randn(batch_size, 1)
        labels = torch.full((batch_size,), -1)
        
        batch = ((time_series, global_features), labels)
        loss = model.training_step(batch, 0)
        
        assert loss.item() >= 0

    def test_validation_step(self):
        """Test validation step."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 784)
        global_features = torch.randn(batch_size, 1)
        labels = torch.randint(0, 10, (batch_size,))
        
        batch = ((time_series, global_features), labels)
        result = model.validation_step(batch, 0)
        
        assert result is not None


class TestLightningAutoencoderOptimizer:
    """Test optimizer configuration."""

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        params = get_basic_params()
        model = LightningAutoencoder(params)
        config = model.configure_optimizers()
        
        assert "optimizer" in config
        assert "lr_scheduler" in config
        assert config["lr_scheduler"]["monitor"] == "val_F1"

    def test_optimizer_learning_rate(self):
        """Test optimizer uses correct learning rate."""
        params = get_basic_params()
        params["LearningRate"] = 0.005
        model = LightningAutoencoder(params)
        config = model.configure_optimizers()
        
        assert config["optimizer"].param_groups[0]["lr"] == 0.005



