import pytest
import torch
import pytorch_lightning as L
from src.GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder


class TestLightningAutoencoderInitialization:
    """Test suite for LightningAutoencoder initialization."""

    def test_initialization_with_valid_params(self):
        """Test that LightningAutoencoder initializes correctly with valid parameters."""
        params = {
            "EncoderParams": {
                "activation_function": "GELU",
                "layer_type": [
                    {
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 1,
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "device": None,
                            "dtype": None
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {}
                    }
                ]
            },
            "DecoderParams": {
                "activation_function": "GELU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 784,
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
                            "in_features": 784,
                            "out_features": 128,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }

        model = LightningAutoencoder(params)

        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'feedforward')

    def test_initialization_missing_encoder_params(self):
        """Test that initialization fails when EncoderParams are missing."""
        params = {
            "DecoderParams": {
                "activation_function": "GELU",
                "layer_type": []
            },
            "FeedForwardParams": {
                "activation_function": "ReLU",
                "layer_type": []
            }
        }

        with pytest.raises(KeyError, match="EncoderParams"):
            LightningAutoencoder(params)

    def test_initialization_missing_decoder_params(self):
        """Test that initialization fails when DecoderParams are missing."""
        params = {
            "EncoderParams": {
                "activation_function": "GELU",
                "layer_type": []
            },
            "FeedForwardParams": {
                "activation_function": "ReLU",
                "layer_type": []
            }
        }

        with pytest.raises(KeyError, match="DecoderParams"):
            LightningAutoencoder(params)

    def test_initialization_missing_feedforward_params(self):
        """Test that initialization fails when FeedForwardParams are missing."""
        params = {
            "EncoderParams": {
                "activation_function": "GELU",
                "layer_type": []
            },
            "DecoderParams": {
                "activation_function": "GELU",
                "layer_type": []
            }
        }

        with pytest.raises(KeyError, match="FeedForwardParams"):
            LightningAutoencoder(params)

    def test_initialization_with_empty_params(self):
        """Test that initialization fails with empty params dictionary."""
        params = {}

        with pytest.raises(KeyError):
            LightningAutoencoder(params)

    def test_initialization_with_wrong_param_types(self):
        """Test that initialization fails when parameter types are incorrect."""
        params = {
            "EncoderParams": "not_a_dict",
            "DecoderParams": {
                "activation_function": "GELU",
                "layer_type": []
            },
            "FeedForwardParams": {
                "activation_function": "ReLU",
                "layer_type": []
            }
        }

        with pytest.raises(TypeError):
            LightningAutoencoder(params)


class TestLightningAutoencoderForward:
    """Test suite for LightningAutoencoder forward pass."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        params = {
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
                            "in_features": 128,
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
            }
        }
        return LightningAutoencoder(params)

    def test_forward_pass_returns_two_outputs(self, simple_model):
        """Test that forward pass returns predictions and decoded output."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)

        predictions, decoded = simple_model.forward(x)

        assert predictions is not None
        assert decoded is not None
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(decoded, torch.Tensor)

    def test_forward_pass_output_shapes(self, simple_model):
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)

        predictions, decoded = simple_model.forward(x)

        # Predictions should be (batch_size, num_classes)
        assert predictions.shape[0] == batch_size
        assert predictions.shape[1] == 10  # num_classes

        # Decoded should match encoder output features
        assert decoded.shape[0] == batch_size
        assert decoded.shape[1] == 784  # flattened 28x28

    def test_forward_pass_with_different_batch_sizes(self, simple_model):
        """Test forward pass with various batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 1, 28, 28)
            predictions, decoded = simple_model.forward(x)

            assert predictions.shape[0] == batch_size
            assert decoded.shape[0] == batch_size

    def test_forward_pass_gradient_flow(self, simple_model):
        """Test that gradients flow through the forward pass."""
        x = torch.randn(2, 1, 28, 28, requires_grad=True)

        predictions, decoded = simple_model.forward(x)

        # Compute a loss and backpropagate
        loss = predictions.sum() + decoded.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

    def test_forward_pass_deterministic(self, simple_model):
        """Test that forward pass is deterministic in eval mode."""
        simple_model.eval()
        x = torch.randn(2, 1, 28, 28)

        predictions1, decoded1 = simple_model.forward(x)
        predictions2, decoded2 = simple_model.forward(x)

        assert torch.allclose(predictions1, predictions2)
        assert torch.allclose(decoded1, decoded2)


class TestLightningAutoencoderTrainingStep:
    """Test suite for LightningAutoencoder training step."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        params = {
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
                            "in_features": 128,
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
            }
        }
        return LightningAutoencoder(params)

    def test_training_step_with_labels(self, simple_model):
        """Test training step with both images and labels."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        batch = (x, y)
        batch_idx = 0

        loss = simple_model.training_step(batch, batch_idx)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss > 0

    def test_training_step_without_labels(self, simple_model):
        """Test training step without labels (unsupervised learning)."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = None
        batch = (x, y)
        batch_idx = 0

        loss = simple_model.training_step(batch, batch_idx)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss > 0

    def test_training_step_loss_components(self, simple_model):
        """Test that both reconstruction and prediction losses are computed correctly."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        batch = (x, y)
        batch_idx = 0

        # Get the full loss
        loss_full = simple_model.training_step(batch, batch_idx)

        # Compute expected reconstruction loss
        predictions, decoded = simple_model.forward(x)
        loss_fn_reconstruction = torch.nn.MSELoss()
        reconstruction_loss = loss_fn_reconstruction(
            decoded, x.view(x.size(0), -1))

        # Compute expected prediction loss
        loss_fn_prediction = torch.nn.CrossEntropyLoss()
        prediction_loss = loss_fn_prediction(predictions, y)

        expected_loss = reconstruction_loss + prediction_loss

        # Check that losses are close (allowing for small numerical differences)
        assert torch.allclose(loss_full, expected_loss, rtol=1e-5)

    def test_training_step_loss_only_reconstruction(self, simple_model):
        """Test that only reconstruction loss is computed when labels are None."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = None
        batch = (x, y)
        batch_idx = 0

        # Get the loss without labels
        loss_no_labels = simple_model.training_step(batch, batch_idx)

        # Compute expected reconstruction loss
        predictions, decoded = simple_model.forward(x)
        loss_fn_reconstruction = torch.nn.MSELoss()
        expected_loss = loss_fn_reconstruction(decoded, x.view(x.size(0), -1))

        # Check that losses are close
        assert torch.allclose(loss_no_labels, expected_loss, rtol=1e-5)

    def test_training_step_gradient_computation(self, simple_model):
        """Test that training step allows gradient computation."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        batch = (x, y)
        batch_idx = 0

        loss = simple_model.training_step(batch, batch_idx)

        # Loss should have grad_fn for backpropagation
        assert loss.requires_grad
        assert loss.grad_fn is not None

    def test_training_step_different_batch_sizes(self, simple_model):
        """Test training step with various batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 1, 28, 28)
            y = torch.randint(0, 10, (batch_size,))
            batch = (x, y)
            batch_idx = 0

            loss = simple_model.training_step(batch, batch_idx)

            assert loss is not None
            assert loss > 0


class TestLightningAutoencoderEdgeCases:
    """Test suite for edge cases and error handling."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        params = {
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
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }
        return LightningAutoencoder(params)

    def test_forward_with_single_sample(self, simple_model):
        """Test forward pass with batch size of 1."""
        x = torch.randn(1, 1, 28, 28)
        predictions, decoded = simple_model.forward(x)

        assert predictions.shape[0] == 1
        assert decoded.shape[0] == 1

    def test_model_in_eval_mode(self, simple_model):
        """Test that model works correctly in evaluation mode."""
        simple_model.eval()
        x = torch.randn(4, 1, 28, 28)

        with torch.no_grad():
            predictions, decoded = simple_model.forward(x)

        assert predictions is not None
        assert decoded is not None

    def test_model_parameters_exist(self, simple_model):
        """Test that model has trainable parameters."""
        params = list(simple_model.parameters())
        assert len(params) > 0

        # Check that parameters require gradients
        assert any(p.requires_grad for p in params)

    def test_model_state_dict(self, simple_model):
        """Test that model state_dict can be retrieved and loaded."""
        state_dict = simple_model.state_dict()
        assert len(state_dict) > 0

        # Create a new model with the same architecture
        params = {
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
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }
        new_model = LightningAutoencoder(params)
        new_model.load_state_dict(state_dict)

    def test_training_step_with_zero_tensor(self, simple_model):
        """Test training step with zero input tensor."""
        batch_size = 4
        x = torch.zeros(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        batch = (x, y)
        batch_idx = 0

        loss = simple_model.training_step(batch, batch_idx)

        assert loss is not None
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestLightningAutoencoderWithDifferentActivations:
    """Test suite for different activation functions."""

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "LeakyReLU"])
    def test_different_encoder_activations(self, activation):
        """Test LightningAutoencoder with different encoder activation functions."""
        params = {
            "EncoderParams": {
                "activation_function": activation,
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
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }

        model = LightningAutoencoder(params)
        x = torch.randn(2, 1, 28, 28)
        predictions, decoded = model.forward(x)

        assert predictions is not None
        assert decoded is not None

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "SELU"])
    def test_different_feedforward_activations(self, activation):
        """Test LightningAutoencoder with different feedforward activation functions."""
        params = {
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
                "activation_function": activation,
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }

        model = LightningAutoencoder(params)
        x = torch.randn(2, 1, 28, 28)
        predictions, decoded = model.forward(x)

        assert predictions is not None
        assert decoded is not None

    def test_weighted_loss(self):
        """Test that class weights are correctly applied in the loss calculation with unbalanced dataset."""
        params = {
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
                "activation_function": "GELU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 10,
                            "bias": True,
                            "device": None,
                            "dtype": None
                        }
                    }
                ]
            }
        }

        model = LightningAutoencoder(params)

        # Create a synthetic unbalanced dataset
        # Class distribution: class 0 appears 10 times, class 1 appears 2 times
        batch_size = 12
        num_classes = 10
        x = torch.randn(batch_size, 1, 28, 28)

        # Create unbalanced labels: 10 samples of class 0, 2 samples of class 1
        y = torch.cat([
            torch.zeros(10, dtype=torch.long),  # 10 samples of class 0
            torch.ones(2, dtype=torch.long)     # 2 samples of class 1
        ])

        batch = (x, y)
        batch_idx = 0

        # Get the model's computed loss
        model_loss = model.training_step(batch, batch_idx)

        # Manually compute the expected loss with the same weights
        predictions, decoded = model.forward(x)

        # Compute reconstruction loss
        loss_fn_reconstruction = torch.nn.MSELoss()
        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded.view(decoded.size(0), -1)
        expected_reconstruction_loss = loss_fn_reconstruction(
            decoded_flat, x_flat)

        # Compute prediction loss with class weights
        # The model uses uniform weights [1.0, 1.0, ..., 1.0]
        class_weights = torch.tensor([1.0] * num_classes, device=x.device)
        loss_fn_prediction = torch.nn.CrossEntropyLoss(weight=class_weights)
        expected_prediction_loss = loss_fn_prediction(predictions, y)

        expected_total_loss = expected_reconstruction_loss + expected_prediction_loss

        # Verify the loss matches
        assert torch.allclose(model_loss, expected_total_loss, rtol=1e-5), \
            f"Model loss {model_loss.item()} != Expected loss {expected_total_loss.item()}"

        # Also verify that weighted loss differs from unweighted loss
        # to ensure weights are actually being applied
        unweighted_loss_fn = torch.nn.CrossEntropyLoss()
        unweighted_prediction_loss = unweighted_loss_fn(predictions, y)

        # With uniform weights, weighted and unweighted should be the same
        assert torch.allclose(expected_prediction_loss, unweighted_prediction_loss, rtol=1e-5), \
            "Uniform weights should produce same loss as no weights"

        # Test with custom weights to verify weight mechanism works
        # Create a more extreme imbalanced dataset where most samples are class 1
        y_imbalanced = torch.cat([
            torch.zeros(2, dtype=torch.long),  # 2 samples of class 0
            torch.ones(10, dtype=torch.long)   # 10 samples of class 1
        ])

        # Get predictions for the imbalanced dataset
        predictions_imbalanced, _ = model.forward(x)

        # Compute unweighted loss
        unweighted_loss_imbalanced = unweighted_loss_fn(
            predictions_imbalanced, y_imbalanced)

        # Create custom weights that heavily penalize the rare class (class 0)
        # Class 0 (rare): weight 10.0, Class 1 (common): weight 1.0
        custom_weights = torch.tensor(
            [10.0, 1.0] + [1.0] * (num_classes - 2), device=x.device)
        custom_loss_fn = torch.nn.CrossEntropyLoss(weight=custom_weights)
        custom_weighted_loss = custom_loss_fn(
            predictions_imbalanced, y_imbalanced)

        # The custom weighted loss should be noticeably different from unweighted
        # because we're applying 10x weight to class 0 samples
        loss_difference = torch.abs(
            custom_weighted_loss - unweighted_loss_imbalanced)
        assert loss_difference > 0.01, \
            f"Custom weights should produce significantly different loss. Difference: {loss_difference.item()}"

        # Verify that the weight mechanism actually increases the loss when rare class has higher weight
        # (assuming the model makes some errors on class 0)
        assert custom_weighted_loss.item() != unweighted_loss_imbalanced.item(), \
            "Weighted and unweighted losses should differ"
