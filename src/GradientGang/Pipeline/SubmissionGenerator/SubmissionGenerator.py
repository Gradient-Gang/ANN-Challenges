import torch
import pandas as pd
import pytorch_lightning as L
from typing import Union, Dict


class SubmissionGenerator:
    """
    Generates submission CSV files for classification challenges.

    Takes a trained model and dataloader, performs predictions, and creates
    a CSV file in the required format with sample indices and predicted labels.
    """

    def __init__(
        self,
        model: L.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        label_mapping: Dict[int, str] = None
    ):
        """
        Initialize the SubmissionGenerator.

        Args:
            model: A trained PyTorch Lightning model
            dataloader: DataLoader containing the test data
            label_mapping: Dictionary mapping integer labels to string labels
                          Default: {0: "no_pain", 1: "low_pain", 2: "high_pain"}
        """
        self.model = model
        self.dataloader = dataloader

        # Reverse label mapping (int -> string)
        if label_mapping is None:
            self.label_mapping = {
                0: "no_pain",
                1: "low_pain",
                2: "high_pain"
            }
        else:
            self.label_mapping = label_mapping

        # Set model to evaluation mode
        self.model.eval()

    def generate_predictions(self) -> torch.Tensor:
        """
        Generate predictions for all samples in the dataloader.

        Returns:
            Tensor of predicted class indices
        """
        all_predictions = []

        with torch.no_grad():
            for batch in self.dataloader:
                # Handle batch with or without labels
                if isinstance(batch, (list, tuple)):
                    features = batch[0]
                else:
                    features = batch

                # Move features to the same device as model
                try:
                    device = next(self.model.parameters()).device
                    features = features.to(device)
                except StopIteration:
                    # Model has no parameters, use CPU
                    pass

                # Get model predictions
                # Handle different model output formats
                outputs = self.model(features)

                # If model returns tuple (predictions, decoded), take predictions
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get predicted class indices
                predicted_classes = torch.argmax(logits, dim=1)
                all_predictions.append(predicted_classes.cpu())

        # Concatenate all predictions
        if len(all_predictions) > 0:
            all_predictions = torch.cat(all_predictions, dim=0)
        else:
            # Return empty tensor for empty dataloader
            all_predictions = torch.tensor([], dtype=torch.long)
        return all_predictions

    def create_submission_file(
        self,
        output_path: str,
        predictions: torch.Tensor = None
    ) -> pd.DataFrame:
        """
        Create a submission CSV file with predictions.

        Args:
            output_path: Path where the submission CSV will be saved
            predictions: Optional pre-computed predictions tensor.
                        If None, predictions will be generated.

        Returns:
            DataFrame containing the submission data
        """
        # Generate predictions if not provided
        if predictions is None:
            predictions = self.generate_predictions()

        # Convert tensor to numpy for easier manipulation
        predictions_np = predictions.numpy()

        # Create sample indices with zero-padding (000, 001, 002, etc.)
        num_samples = len(predictions_np)
        sample_indices = [f"{i:03d}" for i in range(num_samples)]

        # Map integer predictions to string labels
        predicted_labels = [self.label_mapping[pred]
                            for pred in predictions_np]

        # Create DataFrame
        submission_df = pd.DataFrame({
            'sample_index': sample_indices,
            'label': predicted_labels
        })

        # Save to CSV
        submission_df.to_csv(output_path, index=False)

        return submission_df

    def generate_submission(
        self,
        output_path: str = "submission.csv"
    ) -> pd.DataFrame:
        """
        Complete workflow: generate predictions and create submission file.

        Args:
            output_path: Path where the submission CSV will be saved
                        Default: "submission.csv"

        Returns:
            DataFrame containing the submission data
        """

        #  generate submission file
        return self.create_submission_file(output_path)


# Convenience function for quick usage
def generate_submission(
    model: L.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    output_path: str = "submission.csv",
    label_mapping: Dict[int, str] = None
) -> pd.DataFrame:
    """
    Convenience function to generate submission file in one call.

    Args:
        model: A trained PyTorch Lightning model
        dataloader: DataLoader containing the test data
        output_path: Path where the submission CSV will be saved
        label_mapping: Dictionary mapping integer labels to string labels

    Returns:
        DataFrame containing the submission data

    Example:
        >>> from GradientGang.Pipeline.SubmissionGenerator import generate_submission
        >>> submission_df = generate_submission(
        ...     model=trained_model,
        ...     dataloader=test_dataloader,
        ...     output_path="my_submission.csv"
        ... )
    """

    # Initialize SubmissionGenerator 
    generator = SubmissionGenerator(model, dataloader, label_mapping)

    # Generate the submission file
    return generator.generate_submission(output_path)
