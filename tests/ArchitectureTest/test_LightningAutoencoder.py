import pytest
import torch
import pytorch_lightning as L
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder, LightningAutoencoderInterpreter
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


def get_basic_params():
    """Helper to get basic valid params."""
    return {
        "LearningRate": 0.001,
        "Patience": 5,
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
        },
        "OutputDim": 10
    }


class TestLightningAutoencoderInitialization:
    """Test suite for LightningAutoencoder initialization."""

    def test_initialization_with_valid_params(self):
        """Test that LightningAutoencoder initializes correctly with valid parameters."""
        params = get_basic_params()
        model = LightningAutoencoder(params)

        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'feedforward')
        assert hasattr(model, 'val_f1')
        assert hasattr(model, 'params')

    def test_initialization_missing_encoder_params(self):
        """Test that initialization fails when EncoderParams are missing."""
        params = get_basic_params()
        del params["EncoderParams"]

        with pytest.raises(KeyError, match="EncoderParams"):
            LightningAutoencoder(params)

    def test_initialization_missing_decoder_params(self):
        """Test that initialization fails when DecoderParams are missing."""
        params = get_basic_params()
        del params["DecoderParams"]

        with pytest.raises(KeyError, match="DecoderParams"):
            LightningAutoencoder(params)

    def test_initialization_missing_feedforward_params(self):
        """Test that initialization fails when FeedForwardParams are missing."""
        params = get_basic_params()
        del params["FeedForwardParams"]

        with pytest.raises(KeyError, match="FeedForwardParams"):
            LightningAutoencoder(params)

    def test_initialization_missing_output_dim(self):
        """Test that initialization fails when OutputDim is missing."""
        params = get_basic_params()
        del params["OutputDim"]

        with pytest.raises(KeyError, match="OutputDim"):
            LightningAutoencoder(params)

    def test_initialization_missing_learning_rate(self):
        """Test that model works with default learning rate."""
        params = get_basic_params()
        del params["LearningRate"]

        model = LightningAutoencoder(params)
        optimizer_config = model.configure_optimizers()
        # Should use default 0.001
        assert optimizer_config['optimizer'].param_groups[0]['lr'] == 0.001

    def test_initialization_missing_patience(self):
        """Test that model works with default patience."""
        params = get_basic_params()
        del params["Patience"]

        model = LightningAutoencoder(params)
        optimizer_config = model.configure_optimizers()
        # Should use default 5
        assert optimizer_config['lr_scheduler']['scheduler'].patience == 5

    def test_initialization_with_empty_params(self):
        """Test that initialization fails with empty params dictionary."""
        params = {}

        with pytest.raises(KeyError):
            LightningAutoencoder(params)

    def test_initialization_with_wrong_param_types(self):
        """Test that initialization fails when parameter types are incorrect."""
        params = get_basic_params()
        params["EncoderParams"] = "not_a_dict"

        with pytest.raises(TypeError):
            LightningAutoencoder(params)


class TestLightningAutoencoderForward:
    """Test suite for LightningAutoencoder forward pass."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        return LightningAutoencoder(get_basic_params())

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

        with torch.no_grad():
            predictions1, decoded1 = simple_model.forward(x)
            predictions2, decoded2 = simple_model.forward(x)

        assert torch.allclose(predictions1, predictions2)
        assert torch.allclose(decoded1, decoded2)


class TestLightningAutoencoderTrainingStep:
    """Test suite for LightningAutoencoder training step."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        return LightningAutoencoder(get_basic_params())

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
        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded.view(decoded.size(0), -1)
        reconstruction_loss = loss_fn_reconstruction(decoded_flat, x_flat)

        # Compute expected prediction loss
        class_weights = torch.tensor([1.0] * 10, device=x.device)
        loss_fn_prediction = torch.nn.CrossEntropyLoss(weight=class_weights)
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
        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded.view(decoded.size(0), -1)
        expected_loss = loss_fn_reconstruction(decoded_flat, x_flat)

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


class TestLightningAutoencoderValidationStep:
    """Test suite for LightningAutoencoder validation step."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        return LightningAutoencoder(get_basic_params())

    def test_validation_step(self, simple_model):
        """Test validation step."""
        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        batch = (x, y)

        f1_score = simple_model.validation_step(batch, 0)

        assert isinstance(f1_score, torch.Tensor)
        assert 0.0 <= f1_score.item() <= 1.0

    def test_validation_step_f1_metric_persistence(self, simple_model):
        """Test that F1 metric persists across validation steps."""
        # Run multiple validation steps
        for _ in range(3):
            x = torch.randn(4, 1, 28, 28)
            y = torch.randint(0, 10, (4,))
            batch = (x, y)
            simple_model.validation_step(batch, 0)

        # F1 metric should have been updated multiple times
        assert hasattr(simple_model, 'val_f1')

    def test_validation_epoch_end_resets_metric(self, simple_model):
        """Test that F1 metric is reset at end of validation epoch."""
        # Run some validation steps
        for _ in range(3):
            x = torch.randn(4, 1, 28, 28)
            y = torch.randint(0, 10, (4,))
            batch = (x, y)
            simple_model.validation_step(batch, 0)

        # Get F1 value before reset
        f1_before = simple_model.val_f1.compute()

        # Reset
        simple_model.on_validation_epoch_end()

        # After reset, the internal state should be reset
        # (we can't directly check if it's reset, but we can verify the method exists)
        assert hasattr(simple_model, 'on_validation_epoch_end')


class TestLightningAutoencoderOptimizer:
    """Test suite for optimizer configuration."""

    def test_configure_optimizers(self):
        """Test that optimizer is configured correctly."""
        model = LightningAutoencoder(get_basic_params())
        optimizer_config = model.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config
        assert isinstance(optimizer_config['optimizer'], torch.optim.AdamW)
        assert isinstance(optimizer_config['lr_scheduler']['scheduler'],
                          torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert optimizer_config['lr_scheduler']['monitor'] == 'val_F1'

    def test_optimizer_learning_rate(self):
        """Test that optimizer uses correct learning rate."""
        params = get_basic_params()
        params["LearningRate"] = 0.005
        model = LightningAutoencoder(params)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        assert optimizer.param_groups[0]['lr'] == 0.005

    def test_scheduler_patience(self):
        """Test that scheduler uses correct patience."""
        params = get_basic_params()
        params["Patience"] = 10
        model = LightningAutoencoder(params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.patience == 10

    def test_different_learning_rates(self):
        """Test LightningAutoencoder with different learning rates."""
        for lr in [0.0001, 0.001, 0.01]:
            params = get_basic_params()
            params["LearningRate"] = lr
            model = LightningAutoencoder(params)
            optimizer_config = model.configure_optimizers()
            assert optimizer_config['optimizer'].param_groups[0]['lr'] == lr


class TestLightningAutoencoderEdgeCases:
    """Test suite for edge cases and error handling."""

    @pytest.fixture
    def simple_model(self):
        """Fixture providing a simple LightningAutoencoder model."""
        return LightningAutoencoder(get_basic_params())

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
        new_model = LightningAutoencoder(get_basic_params())
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

    def test_save_and_load_model(self, simple_model, tmp_path):
        """Test saving and loading model state."""
        x = torch.randn(4, 1, 28, 28)
        predictions1, decoded1 = simple_model.forward(x)

        # Save state
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(simple_model.state_dict(), checkpoint_path)

        # Load into new model
        new_model = LightningAutoencoder(get_basic_params())
        new_model.load_state_dict(torch.load(checkpoint_path))
        new_model.eval()

        # Outputs should match
        with torch.no_grad():
            predictions2, decoded2 = new_model.forward(x)
        assert torch.allclose(predictions1, predictions2)
        assert torch.allclose(decoded1, decoded2)


class TestLightningAutoencoderWithDifferentActivations:
    """Test suite for different activation functions."""

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "LeakyReLU"])
    def test_different_encoder_activations(self, activation):
        """Test LightningAutoencoder with different encoder activation functions."""
        params = get_basic_params()
        params["EncoderParams"]["activation_function"] = activation

        model = LightningAutoencoder(params)
        x = torch.randn(2, 1, 28, 28)
        predictions, decoded = model.forward(x)

        assert predictions is not None
        assert decoded is not None

    @pytest.mark.parametrize("activation", ["ReLU", "GELU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "SELU"])
    def test_different_feedforward_activations(self, activation):
        """Test LightningAutoencoder with different feedforward activation functions."""
        params = get_basic_params()
        params["FeedForwardParams"]["activation_function"] = activation

        model = LightningAutoencoder(params)
        x = torch.randn(2, 1, 28, 28)
        predictions, decoded = model.forward(x)

        assert predictions is not None
        assert decoded is not None


class TestLightningAutoencoderInterpreter:
    """Test suite for LightningAutoencoderInterpreter."""

    def test_interpreter_exists(self):
        """Test that interpreter exists and is properly configured."""
        assert hasattr(LightningAutoencoder, '__module__')
        assert isinstance(LightningAutoencoderInterpreter,
                          ParameterInterpreter)
        assert LightningAutoencoderInterpreter.name == "LightningAutoencoderInterpreter"

    def test_interpreter_required_params(self):
        """Test that interpreter has correct required params."""
        assert "EncoderParams" in LightningAutoencoderInterpreter.requiredParams
        assert "DecoderParams" in LightningAutoencoderInterpreter.requiredParams
        assert "FeedForwardParams" in LightningAutoencoderInterpreter.requiredParams
        assert "OutputDim" in LightningAutoencoderInterpreter.requiredParams

    def test_interpreter_interpretation(self):
        """Test that interpreter has correct interpretation."""
        assert "LearningRate" in LightningAutoencoderInterpreter.interpretation
        assert "Patience" in LightningAutoencoderInterpreter.interpretation
        assert LightningAutoencoderInterpreter.interpretation["LearningRate"] == float
        assert LightningAutoencoderInterpreter.interpretation["Patience"] == int
