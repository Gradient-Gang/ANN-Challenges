import torch
import pandas as pd
import pytorch_lightning as L
from typing import Union, Dict, Literal
import os
import numpy as np


class WindowedSubmissionGenerator:
    """
    Enhanced SubmissionGenerator that handles windowed predictions.

    When a model is trained with windowing, test samples are also windowed,
    producing multiple predictions per sample. This class aggregates those
    predictions intelligently.
    """

    def __init__(
        self,
        model: L.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        label_mapping: Dict[int, str] = None,
        aggregation_method: Literal[
            "majority_vote", "avg_probs", "max_confidence"
        ] = "avg_probs",
        num_original_samples: int = None,
    ):
        """
        Initialize the WindowedSubmissionGenerator.

        Args:
            model: A trained PyTorch Lightning model
            dataloader: DataLoader containing the test data (possibly windowed)
            label_mapping: Dictionary mapping integer labels to string labels
                          Default: {0: "no_pain", 1: "low_pain", 2: "high_pain"}
            aggregation_method: How to aggregate windowed predictions:
                - "majority_vote": Most frequent predicted class
                - "avg_probs": Average softmax probabilities, then argmax
                - "max_confidence": Take prediction with highest confidence
            num_original_samples: Number of original samples (before windowing)
                                 If None, assumes no windowing
        """
        self.model = model
        self.dataloader = dataloader
        self.aggregation_method = aggregation_method
        self.num_original_samples = num_original_samples

        # Reverse label mapping (int -> string)
        if label_mapping is None:
            self.label_mapping = {0: "no_pain", 1: "low_pain", 2: "high_pain"}
        else:
            self.label_mapping = label_mapping

        # Set model to evaluation mode
        self.model.eval()

    def generate_predictions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for all samples in the dataloader.

        Returns:
            Tuple of (predicted_classes, softmax_probabilities)
        """
        all_predictions = []
        all_probs = []

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
                    # Handle tuple/list of tensors (e.g., Direct architecture)
                    if isinstance(features, (tuple, list)):
                        features = tuple(
                            f.to(device) if isinstance(f, torch.Tensor) else f
                            for f in features
                        )
                    else:
                        features = features.to(device)
                except StopIteration:
                    # Model has no parameters, use CPU
                    pass

                # Get model predictions
                outputs = self.model(features)

                # If model returns tuple (predictions, decoded), take predictions
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get softmax probabilities
                probs = torch.softmax(logits, dim=1)

                # Get predicted class indices
                predicted_classes = torch.argmax(logits, dim=1)

                all_predictions.append(predicted_classes.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all predictions
        if len(all_predictions) > 0:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_probs = torch.cat(all_probs, dim=0)
        else:
            # Return empty tensors for empty dataloader
            all_predictions = torch.tensor([], dtype=torch.long)
            all_probs = torch.tensor([], dtype=torch.float32)

        return all_predictions, all_probs

    def aggregate_windowed_predictions(
        self, predictions: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate predictions from multiple windows back to original samples.

        Args:
            predictions: Tensor of predicted classes for all windows
            probabilities: Tensor of softmax probabilities for all windows

        Returns:
            Tensor of aggregated predictions for original samples
        """
        if self.num_original_samples is None:
            # No windowing, return as-is
            return predictions

        num_windows = len(predictions)
        windows_per_sample = num_windows // self.num_original_samples

        if windows_per_sample == 1:
            # No windowing or already aggregated
            return predictions

        aggregated_predictions = []

        for sample_idx in range(self.num_original_samples):
            start_idx = sample_idx * windows_per_sample
            end_idx = start_idx + windows_per_sample

            # Get all windows for this sample
            sample_predictions = predictions[start_idx:end_idx]
            sample_probs = probabilities[start_idx:end_idx]

            if self.aggregation_method == "majority_vote":
                # Most frequent predicted class
                values, counts = torch.unique(sample_predictions, return_counts=True)
                aggregated_pred = values[torch.argmax(counts)]

            elif self.aggregation_method == "avg_probs":
                # Average probabilities across windows, then take argmax
                avg_probs = torch.mean(sample_probs, dim=0)
                aggregated_pred = torch.argmax(avg_probs)

            elif self.aggregation_method == "max_confidence":
                # Take the prediction with highest confidence
                max_confidence_idx = torch.argmax(torch.max(sample_probs, dim=1)[0])
                aggregated_pred = sample_predictions[max_confidence_idx]

            else:
                raise ValueError(
                    f"Unknown aggregation method: {self.aggregation_method}"
                )

            aggregated_predictions.append(aggregated_pred)

        return torch.tensor(aggregated_predictions, dtype=torch.long)

    def create_submission_file(
        self,
        output_path: str,
        predictions: torch.Tensor = None,
        probabilities: torch.Tensor = None,
    ) -> pd.DataFrame:
        """
        Create a submission CSV file with predictions.

        Args:
            output_path: Path where the submission CSV will be saved
            predictions: Optional pre-computed predictions tensor
            probabilities: Optional pre-computed probabilities tensor

        Returns:
            DataFrame containing the submission data
        """
        # Generate predictions if not provided
        if predictions is None or probabilities is None:
            predictions, probabilities = self.generate_predictions()

        # Aggregate windowed predictions
        aggregated_predictions = self.aggregate_windowed_predictions(
            predictions, probabilities
        )

        # Convert tensor to numpy for easier manipulation
        predictions_np = aggregated_predictions.numpy()

        # Create sample indices with zero-padding (000, 001, 002, etc.)
        num_samples = len(predictions_np)
        sample_indices = [f"{i:03d}" for i in range(num_samples)]

        # Map integer predictions to string labels
        predicted_labels = [self.label_mapping[pred] for pred in predictions_np]

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"sample_index": sample_indices, "label": predicted_labels}
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save to CSV
        submission_df.to_csv(output_path, index=False)

        return submission_df

    def generate_submission(self, output_path: str = "submission.csv") -> pd.DataFrame:
        """
        Complete workflow: generate predictions and create submission file.

        Args:
            output_path: Path where the submission CSV will be saved
                        Default: "submission.csv"

        Returns:
            DataFrame containing the submission data
        """
        return self.create_submission_file(output_path)
