import pytest
import torch
import pandas as pd
import pytorch_lightning as L
import os
import tempfile
from torch.utils.data import DataLoader, TensorDataset
from GradientGang.Pipeline.SubmissionGenerator.SubmissionGenerator import SubmissionGenerator


class MockModel(L.LightningModule):
    """Mock model for testing that returns predictable outputs."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = torch.nn.Linear(10, num_classes)

    def forward(self, x):
        # Handle tuple input (time_series, global_features)
        if isinstance(x, tuple):
            time_series, global_features = x
            x = time_series
        logits = self.linear(x)
        return logits


class DeterministicModel(L.LightningModule):
    """Deterministic model that always returns the same predictions."""

    def __init__(self, predictions_pattern=None):
        super().__init__()
        self.predictions_pattern = predictions_pattern or [0, 1, 2, 0, 1, 2]

    def forward(self, x):
        # Handle tuple input
        if isinstance(x, tuple):
            x = x[0]
        
        batch_size = x.shape[0]
        num_classes = 3
        logits = torch.zeros(batch_size, num_classes)

        # Set logits based on deterministic pattern
        for i in range(batch_size):
            class_idx = self.predictions_pattern[i % len(self.predictions_pattern)]
            logits[i, class_idx] = 10.0  # High logit for predicted class

        return logits


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with 10 samples."""
    features = torch.randn(10, 10)
    labels = torch.zeros(10, dtype=torch.long)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def mock_tuple_dataloader():
    """Create a mock dataloader with tuple inputs (time_series, global_features)."""
    time_series = torch.randn(10, 10)
    global_features = torch.randn(10, 1)
    labels = torch.zeros(10, dtype=torch.long)
    dataset = TensorDataset(time_series, global_features, labels)
    
    class TupleDataLoader:
        def __init__(self, dataset, batch_size):
            self.dataset = dataset
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                end_idx = min(i + self.batch_size, len(self.dataset))
                batch_ts = torch.stack([self.dataset[j][0] for j in range(i, end_idx)])
                batch_gf = torch.stack([self.dataset[j][1] for j in range(i, end_idx)])
                batch_labels = torch.stack([self.dataset[j][2] for j in range(i, end_idx)])
                yield (batch_ts, batch_gf), batch_labels
    
    return TupleDataLoader(dataset, 4)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockModel(num_classes=3)


@pytest.fixture
def deterministic_model():
    """Create a deterministic model."""
    return DeterministicModel()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestSubmissionGeneratorInitialization:
    """Test suite for SubmissionGenerator initialization."""

    def test_initialization_with_default_label_mapping(self, mock_model, mock_dataloader):
        """Test initialization with default label mapping."""
        generator = SubmissionGenerator(mock_model, mock_dataloader)

        assert generator.model == mock_model
        assert generator.dataloader == mock_dataloader
        assert generator.label_mapping == {0: "no_pain", 1: "low_pain", 2: "high_pain"}

    def test_initialization_with_custom_label_mapping(self, mock_model, mock_dataloader):
        """Test initialization with custom label mapping."""
        custom_mapping = {0: "class_a", 1: "class_b", 2: "class_c"}
        generator = SubmissionGenerator(mock_model, mock_dataloader, custom_mapping)

        assert generator.label_mapping == custom_mapping

    def test_model_set_to_eval_mode(self, mock_model, mock_dataloader):
        """Test that model is set to evaluation mode during initialization."""
        mock_model.train()  # Set to training mode first
        assert mock_model.training

        generator = SubmissionGenerator(mock_model, mock_dataloader)

        assert not generator.model.training  # Should be in eval mode


class TestGeneratePredictions:
    """Test suite for generate_predictions method."""

    def test_generate_predictions_returns_tensor(self, deterministic_model, mock_dataloader):
        """Test that generate_predictions returns a tensor."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert isinstance(predictions, torch.Tensor)
        assert predictions.dtype == torch.long

    def test_generate_predictions_correct_length(self, deterministic_model, mock_dataloader):
        """Test that predictions have correct length matching dataset size."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert len(predictions) == 10

    def test_generate_predictions_with_tuple_input(self, deterministic_model, mock_tuple_dataloader):
        """Test predictions with tuple input (time_series, global_features)."""
        generator = SubmissionGenerator(deterministic_model, mock_tuple_dataloader)
        predictions = generator.generate_predictions()

        assert isinstance(predictions, torch.Tensor)
        assert len(predictions) == 10

    def test_generate_predictions_no_gradients(self, mock_model, mock_dataloader):
        """Test that predictions are generated without gradients."""
        generator = SubmissionGenerator(mock_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert not predictions.requires_grad


class TestCreateSubmissionFile:
    """Test suite for create_submission_file method."""

    def test_create_submission_file_creates_csv(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that submission file is created."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.create_submission_file(output_path)

        assert os.path.exists(output_path)
        assert isinstance(df, pd.DataFrame)

    def test_create_submission_file_correct_format(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that submission file has correct format."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.create_submission_file(output_path)

        # Check column names
        assert list(df.columns) == ['sample_index', 'label']

    def test_create_submission_file_with_precomputed_predictions(self, mock_model, mock_dataloader, temp_output_dir):
        """Test creating submission file with pre-computed predictions."""
        generator = SubmissionGenerator(mock_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        # Pre-compute predictions
        predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        df = generator.create_submission_file(output_path, predictions)

        assert len(df) == 10
        assert df['label'].iloc[0] == "no_pain"
        assert df['label'].iloc[1] == "low_pain"
        assert df['label'].iloc[2] == "high_pain"


class TestGenerateSubmission:
    """Test suite for generate_submission method (complete workflow)."""

    def test_generate_submission_complete_workflow(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test complete submission generation workflow."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        assert os.path.exists(output_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ['sample_index', 'label']
