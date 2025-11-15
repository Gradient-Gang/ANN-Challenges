"""Unit tests for EnsembleModel class."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from GradientGang.Pipeline.Utils.EnsembleModels import EnsembleModel


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Handle both single tensor and tuple input (for ensemble compatibility)
        if isinstance(x, tuple):
            x = x[0]  # Use first element (time series data)
        return self.fc(x)


@pytest.fixture
def simple_models():
    """Create list of simple models for testing."""
    return [SimpleModel() for _ in range(5)]


@pytest.fixture
def uniform_weights():
    """Create uniform weights."""
    return torch.ones(5) / 5


@pytest.fixture
def performance_weights():
    """Create performance-based weights."""
    f1_scores = torch.tensor([0.90, 0.88, 0.89, 0.91, 0.87])
    return f1_scores / f1_scores.sum()


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    return torch.randn(32, 10)


class TestEnsembleModel:
    """Test EnsembleModel class."""

    def test_initialization_uniform_weights(self, simple_models, uniform_weights):
        """Test initialization with uniform weights."""
        ensemble = EnsembleModel(simple_models, uniform_weights)

        assert len(ensemble.models) == 5
        assert ensemble.weights.shape == (5,)
        assert torch.allclose(ensemble.weights, uniform_weights)

    def test_initialization_performance_weights(
        self, simple_models, performance_weights
    ):
        """Test initialization with performance-based weights."""
        ensemble = EnsembleModel(simple_models, performance_weights)

        assert len(ensemble.models) == 5
        assert ensemble.weights.shape == (5,)
        # Weights should be normalized
        assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))

    def test_weight_normalization(self, simple_models):
        """Test that weights are automatically normalized."""
        unnormalized_weights = torch.tensor([0.9, 0.8, 0.85, 0.95, 0.7])
        ensemble = EnsembleModel(simple_models, unnormalized_weights)

        # Check normalization
        assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))
        # Check proportions are preserved
        expected = unnormalized_weights / unnormalized_weights.sum()
        assert torch.allclose(ensemble.weights, expected)

    def test_forward_output_shape(self, simple_models, uniform_weights, sample_batch):
        """Test forward pass output shape."""
        ensemble = EnsembleModel(simple_models, uniform_weights)

        output = ensemble(sample_batch)

        # Should return class probabilities (batch_size, num_classes)
        assert output.shape == (32, 2)
        # Should be valid probabilities (sum to 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(32))

    def test_forward_weighted_voting(self, simple_models, sample_batch):
        """Test that forward uses weighted voting."""
        # Create weights that heavily favor first model
        biased_weights = torch.tensor([0.9, 0.025, 0.025, 0.025, 0.025])
        ensemble = EnsembleModel(simple_models, biased_weights)

        # Get ensemble prediction
        ensemble_output = ensemble(sample_batch)

        # Get first model prediction
        first_model_output = torch.softmax(simple_models[0](sample_batch), dim=1)

        # Ensemble output should be close to first model (due to high weight)
        # Allow some difference due to other models
        assert torch.allclose(ensemble_output, first_model_output, atol=0.2)

    def test_predict_step(self, simple_models, uniform_weights, sample_batch):
        """Test predict_step method."""
        ensemble = EnsembleModel(simple_models, uniform_weights)

        # Create mock batch (Lightning passes tuple)
        batch = (sample_batch, torch.zeros(32, dtype=torch.long))

        predictions = ensemble.predict_step(batch, batch_idx=0)

        # Should return class indices
        assert predictions.shape == (32,)
        assert predictions.dtype == torch.long
        assert predictions.min() >= 0
        assert predictions.max() < 2

    def test_device_management(self, simple_models, uniform_weights):
        """Test that ensemble and models move to same device."""
        ensemble = EnsembleModel(simple_models, uniform_weights)

        # Check CPU initially
        assert ensemble.weights.device == torch.device("cpu")
        for model in ensemble.models:
            params = list(model.parameters())
            if params:
                assert params[0].device == torch.device("cpu")

        # Move to CPU explicitly (no GPU in CI)
        ensemble = ensemble.to("cpu")

        assert ensemble.weights.device == torch.device("cpu")
        for model in ensemble.models:
            params = list(model.parameters())
            if params:
                assert params[0].device == torch.device("cpu")

    def test_eval_mode(self, simple_models, uniform_weights):
        """Test that models are in eval mode during forward."""
        ensemble = EnsembleModel(simple_models, uniform_weights)
        ensemble.eval()

        for model in ensemble.models:
            assert not model.training

    def test_single_model_ensemble(self, uniform_weights, sample_batch):
        """Test ensemble with single model."""
        single_model = [SimpleModel()]
        single_weight = torch.tensor([1.0])

        ensemble = EnsembleModel(single_model, single_weight)
        output = ensemble(sample_batch)

        # Should still work correctly
        assert output.shape == (32, 2)
        assert torch.allclose(output.sum(dim=1), torch.ones(32))

    def test_weights_as_buffer(self, simple_models, uniform_weights):
        """Test that weights are registered as buffer (not parameter)."""
        ensemble = EnsembleModel(simple_models, uniform_weights)

        # Weights should be in buffers
        buffers = dict(ensemble.named_buffers())
        assert "weights" in buffers

        # Weights should NOT be in parameters
        params = dict(ensemble.named_parameters())
        assert "weights" not in params

    def test_deterministic_output(self, simple_models, uniform_weights, sample_batch):
        """Test that ensemble gives deterministic output."""
        ensemble = EnsembleModel(simple_models, uniform_weights)
        ensemble.eval()

        # Get two predictions with same input
        with torch.no_grad():
            output1 = ensemble(sample_batch)
            output2 = ensemble(sample_batch)

        assert torch.allclose(output1, output2)

    def test_empty_models_raises_error(self):
        """Test that empty model list raises error."""
        with pytest.raises(Exception):
            EnsembleModel([], torch.tensor([]))

    def test_weight_model_mismatch_raises_error(self, simple_models):
        """Test that mismatched weights/models raises error."""
        wrong_weights = torch.ones(3)  # Only 3 weights for 5 models

        with pytest.raises(Exception):
            EnsembleModel(simple_models, wrong_weights)
