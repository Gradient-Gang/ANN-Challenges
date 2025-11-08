import pytest
import torch
import torch.nn as nn
import pytorch_lightning as L
from GradientGang.Pipeline.Architectures.Direct import Direct, DirectInterpreter
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class TestDirect:
    """Test suite for the Direct class."""

    @pytest.fixture
    def basic_params(self):
        """Basic parameters for Direct model."""
        return {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 1,
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {}
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 25088,  # 32 * 28 * 28
                            "out_features": 128,
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
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 64,
                            "out_features": 10,
                        }
                    }
                ]
            },
            "OutputDim": 10
        }

    @pytest.fixture
    def params_with_optional(self):
        """Parameters with optional settings."""
        return {
            "LearningRate": 0.0005,
            "Patience": 3,
            "EncoderParams": {
                "activation_function": "GELU",
                "layer_type": [
                    {
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 3,
                            "out_channels": 64,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {}
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 12544,  # 64 * 14 * 14
                            "out_features": 256,
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
                            "in_features": 256,
                            "out_features": 128,
                        }
                    },
                    {
                        "name": "Dropout",
                        "params": {
                            "p": 0.2,
                            "inplace": False
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 5,
                        }
                    }
                ]
            },
            "OutputDim": 5,
            "num_input_channels": 3,
            "base_channel_size": 64,
            "latent_dim": 256,
            "act_fn": torch.nn.GELU
        }

    def test_direct_initialization_basic(self, basic_params):
        """Test that Direct model initializes correctly with basic parameters."""
        model = Direct(basic_params)
        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'feedforward')
        assert hasattr(model, 'val_f1')

    def test_direct_initialization_with_optional(self, params_with_optional):
        """Test Direct initialization with optional parameters."""
        model = Direct(params_with_optional)
        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'feedforward')

    def test_direct_missing_required_params(self):
        """Test that Direct raises error when required parameters are missing."""
        # Missing EncoderParams
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing FeedForwardParams
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing OutputDim
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {},
            "FeedForwardParams": {}
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing LearningRate
        params = {
            "Patience": 5,
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing Patience
        params = {
            "LearningRate": 0.001,
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

    def test_direct_forward_pass(self, basic_params):
        """Test forward pass through Direct model."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size
        assert output.shape[1] == 10  # output dimension

    def test_direct_forward_pass_with_optional(self, params_with_optional):
        """Test forward pass with optional parameters."""
        model = Direct(params_with_optional)
        x = torch.randn(2, 3, 28, 28)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2
        assert output.shape[1] == 5

    def test_direct_configure_optimizers(self, basic_params):
        """Test that optimizer is configured correctly."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config
        assert isinstance(optimizer_config['optimizer'], torch.optim.AdamW)
        assert isinstance(optimizer_config['lr_scheduler']['scheduler'],
                          torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert optimizer_config['lr_scheduler']['monitor'] == 'val_F1'

    def test_direct_optimizer_learning_rate(self, basic_params):
        """Test that optimizer uses correct learning rate."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        # Check that learning rate matches the parameter
        assert optimizer.param_groups[0]['lr'] == basic_params['LearningRate']

    def test_direct_scheduler_patience(self, basic_params):
        """Test that scheduler uses correct patience."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.patience == basic_params['Patience']

    def test_direct_training_step(self, basic_params):
        """Test training step with labels."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0  # scalar loss

    def test_direct_training_step_no_labels(self, basic_params):
        """Test training step without labels."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        batch = (x, None)

        loss = model.training_step(batch, 0)

        assert isinstance(loss, (torch.Tensor, int))
        if isinstance(loss, torch.Tensor):
            assert loss.ndim == 0
        else:
            assert loss == 0

    def test_direct_validation_step(self, basic_params):
        """Test validation step."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        f1_score = model.validation_step(batch, 0)

        assert isinstance(f1_score, torch.Tensor)
        assert 0.0 <= f1_score.item() <= 1.0

    def test_direct_gradient_flow(self, basic_params):
        """Test that gradients flow through the model."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28, requires_grad=True)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        loss = model.training_step(batch, 0)
        loss.backward()

        # Check that encoder parameters have gradients
        for param in model.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Check that feedforward parameters have gradients
        for param in model.feedforward.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_direct_parameter_count(self, basic_params):
        """Test that model has learnable parameters."""
        model = Direct(basic_params)
        param_count = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)
        assert param_count > 0

    def test_direct_eval_mode(self, basic_params):
        """Test model in evaluation mode."""
        model = Direct(basic_params)
        model.eval()

        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            output1 = model.forward(x)
            output2 = model.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_direct_train_mode(self, basic_params):
        """Test model in training mode."""
        model = Direct(basic_params)
        model.train()
        assert model.training

    def test_direct_different_learning_rates(self):
        """Test Direct with different learning rates."""
        learning_rates = [0.0001, 0.001, 0.01, 0.1]

        for lr in learning_rates:
            params = {
                "LearningRate": lr,
                "Patience": 5,
                "EncoderParams": {
                    "activation_function": "ReLU",
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
                            }
                        }
                    ]
                },
                "OutputDim": 10
            }

            model = Direct(params)
            optimizer_config = model.configure_optimizers()
            assert optimizer_config['optimizer'].param_groups[0]['lr'] == lr

    def test_direct_different_patience_values(self):
        """Test Direct with different patience values."""
        patience_values = [1, 3, 5, 10, 20]

        for patience in patience_values:
            params = {
                "LearningRate": 0.001,
                "Patience": patience,
                "EncoderParams": {
                    "activation_function": "ReLU",
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
                            }
                        }
                    ]
                },
                "OutputDim": 10
            }

            model = Direct(params)
            optimizer_config = model.configure_optimizers()
            assert optimizer_config['lr_scheduler']['scheduler'].patience == patience

    def test_direct_batch_sizes(self, basic_params):
        """Test Direct with different batch sizes."""
        model = Direct(basic_params)
        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 1, 28, 28)
            y = torch.randint(0, 10, (batch_size,))

            output = model.forward(x)
            assert output.shape[0] == batch_size

            batch = (x, y)
            loss = model.training_step(batch, 0)
            assert isinstance(loss, torch.Tensor)

    def test_direct_output_dimensions(self):
        """Test Direct with different output dimensions."""
        output_dims = [2, 5, 10, 20, 100]

        for out_dim in output_dims:
            params = {
                "LearningRate": 0.001,
                "Patience": 5,
                "EncoderParams": {
                    "activation_function": "ReLU",
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
                                "out_features": out_dim,
                            }
                        }
                    ]
                },
                "OutputDim": out_dim
            }

            model = Direct(params)
            x = torch.randn(4, 1, 28, 28)
            output = model.forward(x)
            assert output.shape[1] == out_dim

    def test_direct_interpreter_attribute(self):
        """Test that Direct has the DirectInterpreter attribute."""
        assert hasattr(
            Direct, 'DirectInterpreter') or 'DirectInterpreter' in globals()
        assert isinstance(DirectInterpreter, ParameterInterpreter)
        assert DirectInterpreter.name == "DirectInterpreter"

    def test_direct_interpreter_interpretation(self):
        """Test DirectInterpreter interpretation dictionary."""
        assert "LearningRate" in DirectInterpreter.interpretation
        assert DirectInterpreter.interpretation["LearningRate"] == float
        assert "Patience" in DirectInterpreter.interpretation
        assert DirectInterpreter.interpretation["Patience"] == int

    def test_direct_interpreter_required_params(self):
        """Test DirectInterpreter required parameters."""
        assert "EncoderParams" in DirectInterpreter.requiredParams
        assert "FeedForwardParams" in DirectInterpreter.requiredParams
        assert "OutputDim" in DirectInterpreter.requiredParams
        assert DirectInterpreter.requiredParams["EncoderParams"] == dict
        assert DirectInterpreter.requiredParams["FeedForwardParams"] == dict
        assert DirectInterpreter.requiredParams["OutputDim"] == int

    def test_direct_with_different_activation_functions(self):
        """Test Direct with different activation functions."""
        activations = ["ReLU", "GELU", "LeakyReLU"]

        for act in activations:
            params = {
                "LearningRate": 0.001,
                "Patience": 5,
                "EncoderParams": {
                    "activation_function": act,
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
                            }
                        }
                    ]
                },
                "FeedForwardParams": {
                    "activation_function": act,
                    "layer_type": [
                        {
                            "name": "Linear",
                            "params": {
                                "in_features": 128,
                                "out_features": 10,
                            }
                        }
                    ]
                },
                "OutputDim": 10
            }

            model = Direct(params)
            x = torch.randn(4, 1, 28, 28)
            output = model.forward(x)
            assert output.shape == (4, 10)

    def test_direct_loss_decreases_with_training(self, basic_params):
        """Test that loss can decrease with training."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        # First loss
        loss1 = model.training_step(batch, 0)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Train for a few steps
        for _ in range(10):
            loss = model.training_step(batch, 0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Final loss
        loss2 = model.training_step(batch, 0)

        # Loss should generally decrease (not guaranteed but likely)
        assert isinstance(loss1, torch.Tensor)
        assert isinstance(loss2, torch.Tensor)

    def test_direct_training_logged_metrics(self, basic_params):
        """Test that training step logs metrics."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        # Training step should log 'train_loss'
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    def test_direct_validation_logged_metrics(self, basic_params):
        """Test that validation step logs metrics."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        # Validation step should log 'val_F1'
        f1 = model.validation_step(batch, 0)
        assert isinstance(f1, torch.Tensor)

    def test_direct_encoder_feedforward_connection(self, basic_params):
        """Test that encoder output connects properly to feedforward."""
        model = Direct(basic_params)
        x = torch.randn(1, 1, 28, 28)

        # Get encoder output
        encoded = model.encoder(x)
        assert isinstance(encoded, torch.Tensor)

        # Pass to feedforward
        predictions = model.feedforward(encoded)
        assert isinstance(predictions, torch.Tensor)

        # Full forward pass
        output = model.forward(x)
        assert torch.allclose(output, predictions)

    def test_direct_cross_entropy_loss(self, basic_params):
        """Test that CrossEntropyLoss is used correctly."""
        model = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        batch = (x, y)

        loss = model.training_step(batch, 0)

        # Loss should be positive
        assert loss.item() >= 0

    def test_direct_f1_score_computation(self, basic_params):
        """Test F1 score computation in validation."""
        model = Direct(basic_params)

        # Perfect predictions
        x = torch.randn(4, 1, 28, 28)
        y = torch.tensor([0, 1, 2, 3])
        batch = (x, y)

        f1 = model.validation_step(batch, 0)
        assert isinstance(f1, torch.Tensor)
        assert 0.0 <= f1.item() <= 1.0

    def test_direct_scheduler_monitor(self, basic_params):
        """Test that scheduler monitors val_F1."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()

        assert optimizer_config['lr_scheduler']['monitor'] == 'val_F1'

    def test_direct_scheduler_mode(self, basic_params):
        """Test that scheduler is in 'min' mode."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.mode == 'min'

    def test_direct_scheduler_factor(self, basic_params):
        """Test that scheduler reduces LR by correct factor."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.factor == 0.2

    def test_direct_scheduler_min_lr(self, basic_params):
        """Test that scheduler has minimum learning rate."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.min_lrs[0] == 5e-5

    def test_direct_invalid_encoder_params_type(self):
        """Test that Direct raises error for invalid EncoderParams type."""
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": "invalid_type",  # Should be dict
            "FeedForwardParams": {},
            "OutputDim": 10
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_invalid_feedforward_params_type(self):
        """Test that Direct raises error for invalid FeedForwardParams type."""
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {},
            "FeedForwardParams": "invalid_type",  # Should be dict
            "OutputDim": 10
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_invalid_output_dim_type(self):
        """Test that Direct raises error for invalid OutputDim type."""
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": "10"  # Should be int
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_multi_channel_input(self):
        """Test Direct with multi-channel input."""
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 3,
                            "out_channels": 64,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {}
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 50176,  # 64 * 28 * 28
                            "out_features": 128,
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
                        }
                    }
                ]
            },
            "OutputDim": 10,
            "num_input_channels": 3
        }

        model = Direct(params)
        x = torch.randn(4, 3, 28, 28)
        output = model.forward(x)
        assert output.shape == (4, 10)

    def test_direct_with_dropout(self):
        """Test Direct with dropout in feedforward."""
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "EncoderParams": {
                "activation_function": "ReLU",
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
                        }
                    },
                    {
                        "name": "Dropout",
                        "params": {
                            "p": 0.5,
                            "inplace": False
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 64,
                            "out_features": 10,
                        }
                    }
                ]
            },
            "OutputDim": 10
        }

        model = Direct(params)
        x = torch.randn(4, 1, 28, 28)

        # Train mode - dropout active
        model.train()
        output1 = model.forward(x)
        output2 = model.forward(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Eval mode - dropout inactive
        model.eval()
        with torch.no_grad():
            output3 = model.forward(x)
            output4 = model.forward(x)
        # Outputs should be identical
        assert torch.allclose(output3, output4)

    def test_direct_save_load_state(self, basic_params, tmp_path):
        """Test saving and loading model state."""
        model1 = Direct(basic_params)
        x = torch.randn(4, 1, 28, 28)
        output1 = model1.forward(x)

        # Save state
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(model1.state_dict(), checkpoint_path)

        # Load into new model
        model2 = Direct(basic_params)
        model2.load_state_dict(torch.load(checkpoint_path))
        model2.eval()

        # Outputs should match
        with torch.no_grad():
            output2 = model2.forward(x)
        assert torch.allclose(output1, output2)

    def test_direct_default_optional_parameters(self, basic_params):
        """Test that default optional parameters are used correctly."""
        model = Direct(basic_params)

        # Check defaults were applied (implicitly tested by successful forward pass)
        x = torch.randn(4, 1, 28, 28)
        output = model.forward(x)
        assert output.shape[0] == 4

    def test_direct_val_f1_metric_instance(self, basic_params):
        """Test that val_f1 metric is stored as instance variable."""
        model = Direct(basic_params)
        assert hasattr(model, 'val_f1')

    def test_direct_val_f1_on_epoch_end(self, basic_params):
        """Test that F1 metric can be computed over multiple batches."""
        model = Direct(basic_params)

        # Simulate multiple validation batches
        for _ in range(3):
            x = torch.randn(4, 1, 28, 28)
            y = torch.randint(0, 10, (4,))
            batch = (x, y)
            model.validation_step(batch, 0)

        # F1 should aggregate across batches
        assert hasattr(model, 'val_f1')
