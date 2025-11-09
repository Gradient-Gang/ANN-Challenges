import pytest
import torch
import pandas as pd
import pytorch_lightning as L
import os
import tempfile
from torch.utils.data import DataLoader, TensorDataset
from GradientGang.Pipeline.SubmissionGenerator.SubmissionGenerator import SubmissionGenerator, generate_submission


class MockModel(L.LightningModule):
    """Mock model for testing that returns predictable outputs."""

    def __init__(self, num_classes=3, return_tuple=True):
        super().__init__()
        self.linear = torch.nn.Linear(10, num_classes)
        self.return_tuple = return_tuple

    def forward(self, x):
        logits = self.linear(x)
        if self.return_tuple:
            # Simulate autoencoder that returns (predictions, decoded)
            return logits, x
        else:
            # Simulate simple classifier
            return logits


class DeterministicModel(L.LightningModule):
    """Deterministic model that always returns the same predictions."""

    def __init__(self, predictions_pattern=None):
        super().__init__()
        self.predictions_pattern = predictions_pattern or [0, 1, 2, 0, 1, 2]

    def forward(self, x):
        batch_size = x.shape[0]
        num_classes = 3
        logits = torch.zeros(batch_size, num_classes)

        # Set logits based on deterministic pattern
        for i in range(batch_size):
            class_idx = self.predictions_pattern[i % len(
                self.predictions_pattern)]
            logits[i, class_idx] = 10.0  # High logit for predicted class

        return logits, x


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with 10 samples."""
    features = torch.randn(10, 10)
    labels = torch.zeros(10, dtype=torch.long)  # Dummy labels
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockModel(num_classes=3, return_tuple=True)


@pytest.fixture
def simple_model():
    """Create a simple non-tuple model."""
    return MockModel(num_classes=3, return_tuple=False)


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
        assert generator.label_mapping == {
            0: "no_pain", 1: "low_pain", 2: "high_pain"}

    def test_initialization_with_custom_label_mapping(self, mock_model, mock_dataloader):
        """Test initialization with custom label mapping."""
        custom_mapping = {0: "class_a", 1: "class_b", 2: "class_c"}
        generator = SubmissionGenerator(
            mock_model, mock_dataloader, custom_mapping)

        assert generator.label_mapping == custom_mapping

    def test_model_set_to_eval_mode(self, mock_model, mock_dataloader):
        """Test that model is set to evaluation mode during initialization."""
        mock_model.train()  # Set to training mode first
        assert mock_model.training

        generator = SubmissionGenerator(mock_model, mock_dataloader)

        assert not generator.model.training  # Should be in eval mode

    def test_initialization_with_different_num_classes(self, mock_dataloader):
        """Test initialization with models having different number of classes."""
        for num_classes in [2, 3, 5, 10]:
            model = MockModel(num_classes=num_classes)
            generator = SubmissionGenerator(
                model,
                mock_dataloader,
                {i: f"class_{i}" for i in range(num_classes)}
            )
            assert len(generator.label_mapping) == num_classes

    def test_initialization_preserves_model_parameters(self, mock_model, mock_dataloader):
        """Test that initialization doesn't modify model parameters."""
        original_params = {name: param.clone()
                           for name, param in mock_model.named_parameters()}

        generator = SubmissionGenerator(mock_model, mock_dataloader)

        for name, param in generator.model.named_parameters():
            assert torch.allclose(param, original_params[name])


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

        # Mock dataloader has 10 samples
        assert len(predictions) == 10

    def test_generate_predictions_with_tuple_output(self, mock_model, mock_dataloader):
        """Test predictions with model that returns tuple (predictions, decoded)."""
        mock_model.return_tuple = True
        generator = SubmissionGenerator(mock_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert isinstance(predictions, torch.Tensor)
        assert len(predictions) == 10

    def test_generate_predictions_with_single_output(self, simple_model, mock_dataloader):
        """Test predictions with model that returns single tensor."""
        generator = SubmissionGenerator(simple_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert isinstance(predictions, torch.Tensor)
        assert len(predictions) == 10

    def test_generate_predictions_no_gradients(self, mock_model, mock_dataloader):
        """Test that predictions are generated without gradients."""
        generator = SubmissionGenerator(mock_model, mock_dataloader)
        predictions = generator.generate_predictions()

        assert not predictions.requires_grad

    def test_generate_predictions_deterministic(self, deterministic_model, mock_dataloader):
        """Test that predictions are deterministic in eval mode."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)

        predictions1 = generator.generate_predictions()
        predictions2 = generator.generate_predictions()

        assert torch.equal(predictions1, predictions2)

    def test_generate_predictions_with_different_batch_sizes(self, deterministic_model):
        """Test predictions with different batch sizes."""
        for batch_size in [1, 2, 5, 10]:
            features = torch.randn(20, 10)
            labels = torch.zeros(20, dtype=torch.long)
            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False)

            generator = SubmissionGenerator(deterministic_model, dataloader)
            predictions = generator.generate_predictions()

            assert len(predictions) == 20

    def test_generate_predictions_with_empty_dataloader(self, mock_model):
        """Test predictions with empty dataloader."""
        features = torch.randn(0, 10)
        labels = torch.zeros(0, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        generator = SubmissionGenerator(mock_model, dataloader)
        predictions = generator.generate_predictions()

        assert len(predictions) == 0
        assert isinstance(predictions, torch.Tensor)

    def test_generate_predictions_moves_to_model_device(self, mock_model, mock_dataloader):
        """Test that features are moved to the same device as model."""
        # This test verifies the device handling logic
        generator = SubmissionGenerator(mock_model, mock_dataloader)
        predictions = generator.generate_predictions()

        model_device = next(mock_model.parameters()).device
        assert predictions.device == torch.device(
            'cpu')  # Predictions are moved to CPU

    def test_generate_predictions_handles_batch_without_labels(self, mock_model):
        """Test predictions when dataloader returns features without labels."""
        features = torch.randn(10, 10)
        dataset = TensorDataset(features)  # No labels
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        generator = SubmissionGenerator(mock_model, dataloader)
        predictions = generator.generate_predictions()

        assert len(predictions) == 10


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

        # Check data types
        assert df['sample_index'].dtype == object  # String type
        assert df['label'].dtype == object  # String type

    def test_create_submission_file_correct_indices(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that sample indices are correctly zero-padded."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.create_submission_file(output_path)

        # Check index format (000, 001, 002, ...)
        assert df['sample_index'].iloc[0] == "000"
        assert df['sample_index'].iloc[1] == "001"
        assert df['sample_index'].iloc[9] == "009"

    def test_create_submission_file_correct_labels(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that labels are correctly mapped to strings."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.create_submission_file(output_path)

        # Check that all labels are valid strings from the mapping
        valid_labels = {"no_pain", "low_pain", "high_pain"}
        assert all(label in valid_labels for label in df['label'])

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

    def test_create_submission_file_custom_label_mapping(self, mock_model, mock_dataloader, temp_output_dir):
        """Test submission file with custom label mapping."""
        custom_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        generator = SubmissionGenerator(
            mock_model, mock_dataloader, custom_mapping)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        predictions = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 1, 2])
        df = generator.create_submission_file(output_path, predictions)

        assert df['label'].iloc[0] == "negative"
        assert df['label'].iloc[1] == "neutral"
        assert df['label'].iloc[2] == "positive"

    def test_create_submission_file_correct_row_count(self, deterministic_model, temp_output_dir):
        """Test submission file has correct number of rows for different dataset sizes."""
        for num_samples in [5, 10, 50, 100]:
            features = torch.randn(num_samples, 10)
            labels = torch.zeros(num_samples, dtype=torch.long)
            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

            generator = SubmissionGenerator(deterministic_model, dataloader)
            output_path = os.path.join(
                temp_output_dir, f"submission_{num_samples}.csv")

            df = generator.create_submission_file(output_path)

            assert len(df) == num_samples

    def test_create_submission_file_zero_padding_for_large_numbers(self, deterministic_model, temp_output_dir):
        """Test that zero-padding works correctly for larger sample counts."""
        num_samples = 150
        features = torch.randn(num_samples, 10)
        labels = torch.zeros(num_samples, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

        generator = SubmissionGenerator(deterministic_model, dataloader)
        output_path = os.path.join(temp_output_dir, "submission_large.csv")

        df = generator.create_submission_file(output_path)

        # Check that 3-digit padding is used
        assert df['sample_index'].iloc[0] == "000"
        assert df['sample_index'].iloc[99] == "099"
        assert df['sample_index'].iloc[149] == "149"

    def test_create_submission_file_csv_readable(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that created CSV file can be read back correctly."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df_written = generator.create_submission_file(output_path)
        # Read sample_index as string
        df_read = pd.read_csv(output_path, dtype={'sample_index': str})

        pd.testing.assert_frame_equal(df_written, df_read)

    def test_create_submission_file_overwrites_existing(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that existing submission file is overwritten."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        # Create first file
        df1 = generator.create_submission_file(
            output_path, torch.zeros(10, dtype=torch.long))

        # Create second file with different predictions
        df2 = generator.create_submission_file(
            output_path, torch.ones(10, dtype=torch.long))

        # Read the file to verify it was overwritten
        df_read = pd.read_csv(output_path)
        # Should be from second write
        assert df_read['label'].iloc[0] == "low_pain"


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

    def test_generate_submission_default_filename(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test generate_submission with default filename."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)

        # Change to temp directory to test default filename
        original_dir = os.getcwd()
        try:
            os.chdir(temp_output_dir)
            df = generator.generate_submission()

            assert os.path.exists("submission.csv")
        finally:
            os.chdir(original_dir)

    def test_generate_submission_returns_dataframe(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that generate_submission returns a DataFrame."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        result = generator.generate_submission(output_path)

        assert isinstance(result, pd.DataFrame)

    def test_generate_submission_consistent_with_predictions(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that submission file is consistent with generated predictions."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        # Generate predictions separately
        predictions = generator.generate_predictions()

        # Generate submission
        df = generator.generate_submission(output_path)

        # Verify labels match predictions
        for i, pred in enumerate(predictions):
            expected_label = generator.label_mapping[pred.item()]
            assert df['label'].iloc[i] == expected_label


class TestConvenienceFunction:
    """Test suite for the convenience function generate_submission."""

    def test_convenience_function_creates_submission(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that convenience function creates submission file."""
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generate_submission(
            deterministic_model, mock_dataloader, output_path)

        assert os.path.exists(output_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_convenience_function_with_custom_mapping(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test convenience function with custom label mapping."""
        output_path = os.path.join(temp_output_dir, "submission.csv")
        custom_mapping = {0: "cat", 1: "dog", 2: "bird"}

        df = generate_submission(
            deterministic_model, mock_dataloader, output_path, custom_mapping)

        valid_labels = {"cat", "dog", "bird"}
        assert all(label in valid_labels for label in df['label'])

    def test_convenience_function_default_output_path(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test convenience function with default output path."""
        original_dir = os.getcwd()
        try:
            os.chdir(temp_output_dir)
            df = generate_submission(deterministic_model, mock_dataloader)

            assert os.path.exists("submission.csv")
        finally:
            os.chdir(original_dir)

    def test_convenience_function_matches_class_method(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that convenience function produces same result as class method."""
        output_path1 = os.path.join(temp_output_dir, "submission1.csv")
        output_path2 = os.path.join(temp_output_dir, "submission2.csv")

        # Use convenience function
        df1 = generate_submission(
            deterministic_model, mock_dataloader, output_path1)

        # Use class method
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        df2 = generator.generate_submission(output_path2)

        # Compare DataFrames (they should be identical for deterministic model)
        pd.testing.assert_frame_equal(df1, df2)


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_single_sample_dataset(self, deterministic_model, temp_output_dir):
        """Test with dataset containing only one sample."""
        features = torch.randn(1, 10)
        labels = torch.zeros(1, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=1)

        generator = SubmissionGenerator(deterministic_model, dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        assert len(df) == 1
        assert df['sample_index'].iloc[0] == "000"

    def test_large_dataset(self, deterministic_model, temp_output_dir):
        """Test with large dataset to ensure scalability."""
        num_samples = 1000
        features = torch.randn(num_samples, 10)
        labels = torch.zeros(num_samples, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        generator = SubmissionGenerator(deterministic_model, dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        assert len(df) == num_samples
        assert df['sample_index'].iloc[999] == "999"

    def test_model_on_different_device(self, mock_dataloader, temp_output_dir):
        """Test with model and data on CPU (default device)."""
        model = MockModel(num_classes=3)
        model = model.to('cpu')

        generator = SubmissionGenerator(model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        assert len(df) == 10

    def test_dataloader_with_different_feature_dimensions(self, deterministic_model, temp_output_dir):
        """Test with different feature dimensions."""
        for feature_dim in [5, 10, 20, 50]:
            features = torch.randn(10, feature_dim)
            labels = torch.zeros(10, dtype=torch.long)
            dataset = TensorDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=4)

            # Create model with matching input dimension
            model = MockModel(num_classes=3)
            model.linear = torch.nn.Linear(feature_dim, 3)

            generator = SubmissionGenerator(model, dataloader)
            output_path = os.path.join(
                temp_output_dir, f"submission_{feature_dim}.csv")

            df = generator.generate_submission(output_path)

            assert len(df) == 10

    def test_all_predictions_same_class(self, mock_dataloader, temp_output_dir):
        """Test when all predictions are the same class."""
        # Model that always predicts class 0
        model = DeterministicModel(predictions_pattern=[0, 0, 0, 0, 0, 0])

        generator = SubmissionGenerator(model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        # All labels should be "no_pain" (class 0)
        assert all(df['label'] == "no_pain")

    def test_predictions_cover_all_classes(self, mock_dataloader, temp_output_dir):
        """Test that predictions can cover all classes."""
        model = DeterministicModel(predictions_pattern=[0, 1, 2, 0, 1, 2])

        generator = SubmissionGenerator(model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        # Check that all three classes appear
        label_counts = df['label'].value_counts()
        assert len(label_counts) >= 3 or len(df) < 3

    def test_output_path_with_nested_directories(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test creating submission file in nested directory structure."""
        nested_path = os.path.join(
            temp_output_dir, "results", "submissions", "submission.csv")
        os.makedirs(os.path.dirname(nested_path), exist_ok=True)

        generator = SubmissionGenerator(deterministic_model, mock_dataloader)

        df = generator.generate_submission(nested_path)

        assert os.path.exists(nested_path)
        assert len(df) == 10

    def test_label_mapping_with_missing_class(self, mock_dataloader, temp_output_dir):
        """Test behavior when label mapping is missing a class (should raise error)."""
        model = MockModel(num_classes=3)
        incomplete_mapping = {0: "no_pain", 1: "low_pain"}  # Missing class 2

        generator = SubmissionGenerator(
            model, mock_dataloader, incomplete_mapping)

        # Generate predictions that include class 2
        predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        output_path = os.path.join(temp_output_dir, "submission.csv")

        # This should raise a KeyError when trying to map class 2
        with pytest.raises(KeyError):
            generator.create_submission_file(output_path, predictions)


class TestIntegrationWithRealScenarios:
    """Test suite for integration with real-world scenarios."""

    def test_typical_classification_workflow(self, temp_output_dir):
        """Test typical workflow: train model, load test data, generate submission."""
        # Simulate a trained model
        model = DeterministicModel(predictions_pattern=[0, 1, 2, 1, 0])

        # Simulate test dataset
        num_test_samples = 50
        test_features = torch.randn(num_test_samples, 10)
        test_labels = torch.zeros(num_test_samples, dtype=torch.long)
        test_dataset = TensorDataset(test_features, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Generate submission
        output_path = os.path.join(temp_output_dir, "submission.csv")
        df = generate_submission(model, test_dataloader, output_path)

        # Verify submission
        assert len(df) == num_test_samples
        assert list(df.columns) == ['sample_index', 'label']
        assert os.path.exists(output_path)

    def test_submission_format_matches_specification(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that submission format exactly matches the required specification."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        generator.generate_submission(output_path)

        # Read the CSV with proper dtype to preserve string format
        df = pd.read_csv(output_path, dtype={'sample_index': str})

        # Check header
        assert list(df.columns) == ['sample_index', 'label']

        # Check first few rows match format
        assert df['sample_index'].iloc[0] == "000"
        assert df['label'].iloc[0] in ["no_pain", "low_pain", "high_pain"]

        # Check that there are no extra columns
        assert len(df.columns) == 2

        # Check that all indices are zero-padded strings
        for idx in df['sample_index']:
            assert isinstance(idx, str)
            assert len(idx) == 3
            assert idx.isdigit()

    def test_multiple_submissions_from_same_generator(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test generating multiple submission files from same generator."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)

        # Generate multiple submissions
        for i in range(3):
            output_path = os.path.join(temp_output_dir, f"submission_{i}.csv")
            df = generator.generate_submission(output_path)

            assert os.path.exists(output_path)
            assert len(df) == 10


class TestOutputFormat:
    """Test suite specifically for output format validation."""

    def test_csv_has_no_index_column(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that CSV file doesn't include pandas index column."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        generator.generate_submission(output_path)

        # Read CSV and check it has exactly 2 columns
        with open(output_path, 'r') as f:
            first_line = f.readline().strip()
            assert first_line == "sample_index,label"

            second_line = f.readline().strip()
            # Should have format: 000,pain_level (no index column)
            parts = second_line.split(',')
            assert len(parts) == 2

    def test_sample_indices_are_strings(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that sample indices are stored as strings, not integers."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        # Check that sample_index is string type
        assert df['sample_index'].dtype == object

        # Check that indices maintain leading zeros
        assert df['sample_index'].iloc[0] == "000"
        assert df['sample_index'].iloc[0] != "0"

    def test_labels_are_lowercase_with_underscores(self, deterministic_model, mock_dataloader, temp_output_dir):
        """Test that labels follow the correct naming convention."""
        generator = SubmissionGenerator(deterministic_model, mock_dataloader)
        output_path = os.path.join(temp_output_dir, "submission.csv")

        df = generator.generate_submission(output_path)

        # All labels should be lowercase with underscores
        valid_format = all(
            label.islower() and ' ' not in label
            for label in df['label']
        )
        assert valid_format
