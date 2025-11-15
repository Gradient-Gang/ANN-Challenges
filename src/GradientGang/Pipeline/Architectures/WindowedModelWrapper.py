"""
WindowedModelWrapper: Applies windowing at the model level instead of data loading level.

This wrapper allows models to receive full sequences, create windows internally,
and aggregate predictions. This ensures training/validation metrics match inference metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Tuple, Optional, Literal
from torchmetrics import F1Score


class WindowedModelWrapper(L.LightningModule):
    """
    Wrapper that adds windowing capability to any LightningModule.

    Key Features:
    - Receives full sequences (no windowing in DataLoader)
    - Creates windows internally during forward pass
    - Aggregates window predictions to get sample-level predictions
    - Training/validation metrics match inference metrics

    Args:
        base_model (L.LightningModule): The base model (e.g., LightningAutoencoder, Direct)
        window_size (int): Size of each window
        stride (int): Stride for sliding window
        aggregation_method (str): How to aggregate window predictions
            - "avg_probs": Average softmax probabilities (recommended)
            - "avg_logits": Average logits before softmax
            - "majority_vote": Majority voting across windows
            - "max_confidence": Pick prediction with highest confidence
        window_loss_weight (float): Weight for auxiliary window-level loss (0-1)
            - 0.0: Only sample-level loss (no window supervision)
            - 1.0: Only window-level loss (each window supervised)
            - 0.3: 30% window loss, 70% sample loss (recommended)
    """

    def __init__(
        self,
        base_model: L.LightningModule,
        window_size: int = 80,
        stride: int = 40,
        aggregation_method: Literal[
            "avg_probs", "avg_logits", "majority_vote", "max_confidence"
        ] = "avg_probs",
        window_loss_weight: float = 0.3,
    ):
        super().__init__()

        self.base_model = base_model
        self.window_size = window_size
        self.stride = stride
        self.aggregation_method = aggregation_method
        self.window_loss_weight = window_loss_weight

        # Extract num_classes from base model if available
        if hasattr(base_model, "params"):
            self.num_classes = base_model.params.get("OutputDim", 3)
        else:
            self.num_classes = 3  # Default for pirate pain dataset

        # Initialize F1 metric for sample-level predictions
        self.f1_metric = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["base_model"])

    def create_windows(self, time_series: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Create windows from full time series sequences using efficient PyTorch operations.

        This uses torch.unfold which is highly optimized for sliding window operations.

        Args:
            time_series: [batch, features, seq_len] full sequences

        Returns:
            windows: [batch, num_windows, features, window_size]
            num_windows: Number of windows created per sample
        """
        batch_size, features, seq_len = time_series.shape

        # Calculate number of windows
        num_windows = ((seq_len - self.window_size) // self.stride) + 1

        # Fast path: if sequence length is exactly divisible, use unfold (fastest)
        if (
            seq_len - self.window_size
        ) % self.stride == 0 and seq_len >= self.window_size:
            # unfold creates sliding windows along dimension 2 (time)
            # Result: [batch, features, num_windows, window_size]
            windows = time_series.unfold(
                dimension=2, size=self.window_size, step=self.stride
            )
            # Transpose to desired shape: [batch, num_windows, features, window_size]
            windows = windows.permute(0, 2, 1, 3).contiguous()

        else:
            # Slow path: need padding (rare case)
            # Pad sequence to make it divisible
            pad_size = (num_windows - 1) * self.stride + self.window_size - seq_len
            if pad_size > 0:
                time_series_padded = F.pad(
                    time_series, (0, pad_size), mode="constant", value=0
                )
            else:
                time_series_padded = time_series

            # Now use unfold on padded sequence
            windows = time_series_padded.unfold(
                dimension=2, size=self.window_size, step=self.stride
            )
            windows = windows.permute(0, 2, 1, 3).contiguous()

            # Ensure we have exactly num_windows
            windows = windows[:, :num_windows, :, :]

        return windows, num_windows

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with windowing.

        Args:
            x: Tuple of (time_series, global_features)
                time_series: [batch, features, seq_len] full sequences
                global_features: [batch, global_dim]

        Returns:
            If training with window_loss_weight > 0:
                Tuple of (sample_logits, window_logits)
            Else:
                sample_logits: [batch, num_classes]
        """
        time_series, global_features = x
        batch_size = time_series.shape[0]

        # Create windows: [batch, num_windows, features, window_size]
        windows, num_windows = self.create_windows(time_series)

        # Reshape for base model: [batch * num_windows, features, window_size]
        windows_flat = windows.view(-1, windows.shape[2], windows.shape[3])

        # Expand global features for each window
        # [batch, global_dim] -> [batch * num_windows, global_dim]
        global_features_expanded = (
            global_features.unsqueeze(1)
            .expand(-1, num_windows, -1)
            .reshape(-1, global_features.shape[1])
        )

        # Process all windows through base model
        base_input = (windows_flat, global_features_expanded)

        # Handle different return types (some models return reconstructions)
        base_output = self.base_model(base_input)
        if isinstance(base_output, tuple):
            # LightningAutoencoder returns (predictions, (decoded, decoded_global))
            window_logits_flat = base_output[0]
        else:
            # Direct model returns just predictions
            window_logits_flat = base_output

        # Reshape back: [batch, num_windows, num_classes]
        window_logits = window_logits_flat.view(batch_size, num_windows, -1)

        # Aggregate window predictions to get sample-level prediction
        sample_logits = self.aggregate_predictions(window_logits)

        # During training with auxiliary loss, return both
        if self.training:
            return sample_logits, window_logits
        else:
            return sample_logits

    def aggregate_predictions(self, window_logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate window-level predictions to sample-level predictions.

        Args:
            window_logits: [batch, num_windows, num_classes]

        Returns:
            sample_logits: [batch, num_classes]
        """
        if self.aggregation_method == "avg_probs":
            # Average probabilities (recommended)
            window_probs = F.softmax(window_logits, dim=-1)
            avg_probs = window_probs.mean(dim=1)
            # Use log_softmax for numerical stability and gradient flow
            # Convert avg_probs back to logits: log(p) = log(softmax(x))
            sample_logits = torch.log(avg_probs.clamp(min=1e-8))

        elif self.aggregation_method == "avg_logits":
            # Average logits directly
            sample_logits = window_logits.mean(dim=1)

        elif self.aggregation_method == "majority_vote":
            # Majority voting
            window_preds = window_logits.argmax(dim=-1)  # [batch, num_windows]
            # Convert to one-hot and sum
            vote_counts = (
                F.one_hot(window_preds, num_classes=self.num_classes).sum(dim=1).float()
            )
            # Normalize to pseudo-probabilities
            sample_probs = vote_counts / vote_counts.sum(dim=1, keepdim=True)
            sample_logits = torch.log(sample_probs + 1e-8)

        elif self.aggregation_method == "max_confidence":
            # Pick window with highest confidence
            window_probs = F.softmax(window_logits, dim=-1)
            max_probs, _ = window_probs.max(dim=-1)  # [batch, num_windows]
            best_window_idx = max_probs.argmax(dim=1)  # [batch]

            # Gather logits from best window
            batch_indices = torch.arange(
                window_logits.shape[0], device=window_logits.device
            )
            sample_logits = window_logits[batch_indices, best_window_idx]

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return sample_logits

    def compute_loss(
        self,
        sample_logits: torch.Tensor,
        window_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            sample_logits: [batch, num_classes] aggregated predictions
            window_logits: [batch, num_windows, num_classes] or None
            labels: [batch] ground truth labels

        Returns:
            total_loss, sample_loss, window_loss
        """
        # Filter out unlabeled samples (-1)
        labeled_mask = labels >= 0

        if not labeled_mask.any():
            # No labeled samples in batch - return None to skip
            # Lightning will handle this gracefully
            return None, None, None

        # Sample-level loss (main objective)
        labeled_sample_logits = sample_logits[labeled_mask]
        labeled_targets = labels[labeled_mask]

        # Get class weights from base model if available
        if hasattr(self.base_model, "class_weights"):
            weight = self.base_model.class_weights
        else:
            weight = None

        sample_loss = F.cross_entropy(
            labeled_sample_logits, labeled_targets, weight=weight
        )

        # Window-level loss (auxiliary objective)
        if window_logits is not None and self.window_loss_weight > 0:
            # Each window should also predict the correct class
            labeled_window_logits = window_logits[
                labeled_mask
            ]  # [num_labeled, num_windows, num_classes]
            num_labeled = labeled_window_logits.shape[0]
            num_windows = labeled_window_logits.shape[1]

            # Expand labels for all windows
            window_targets = labeled_targets.unsqueeze(1).expand(
                -1, num_windows
            )  # [num_labeled, num_windows]

            # Flatten and compute loss
            window_logits_flat = labeled_window_logits.view(-1, self.num_classes)
            window_targets_flat = window_targets.reshape(-1)

            window_loss = F.cross_entropy(
                window_logits_flat, window_targets_flat, weight=weight
            )
        else:
            window_loss = torch.tensor(0.0, device=sample_logits.device)

        # Combined loss
        total_loss = (
            1 - self.window_loss_weight
        ) * sample_loss + self.window_loss_weight * window_loss

        return total_loss, sample_loss, window_loss

    def training_step(self, batch, batch_idx):
        """Training step that respects base model's loss computation."""
        (time_series, global_features), labels = batch
        batch_size = time_series.shape[0]

        # Create windows and expand inputs for window-level processing
        windows, num_windows = self.create_windows(time_series)
        windows_flat = windows.view(-1, windows.shape[2], windows.shape[3])
        global_features_expanded = (
            global_features.unsqueeze(1)
            .expand(-1, num_windows, -1)
            .reshape(-1, global_features.shape[1])
        )
        
        # Expand labels for each window (each window should predict same class as sample)
        labels_expanded = labels.unsqueeze(1).expand(-1, num_windows).reshape(-1)
        
        # Create batch for base model with windowed data
        base_input = (windows_flat, global_features_expanded)
        
        # For autoencoders, manually compute losses to respect base model's logic
        if hasattr(self.base_model, 'computeReconstructionLoss') and hasattr(self.base_model, 'computePredictionLoss'):
            # This is a LightningAutoencoder - compute its losses manually
            # Forward pass through base model
            predictions, (reconstructed_timeSeries, reconstructed_globalFeatures) = self.base_model(base_input)
            
            # Compute reconstruction losses
            reconstruction_loss_ts, reconstruction_loss_gf = self.base_model.computeReconstructionLoss(
                windows_flat, global_features_expanded,
                reconstructed_timeSeries, reconstructed_globalFeatures
            )
            reconstruction_loss = reconstruction_loss_ts + reconstruction_loss_gf
            
            # Compute prediction loss
            prediction_loss = self.base_model.computePredictionLoss(labels_expanded, predictions)
            
            # Combined loss using base model's weighting
            reconstruction_weight = self.base_model.reconstruction_loss_weight
            base_loss = reconstruction_weight * reconstruction_loss + (1 - reconstruction_weight) * prediction_loss
            
            # Log base model's internal metrics
            self.log("train_reconstruction_loss_timeSeries", reconstruction_loss_ts)
            self.log("train_reconstruction_loss_globalFeatures", reconstruction_loss_gf)
            self.log("train_reconstruction_loss", reconstruction_loss)
            self.log("train_prediction_loss", prediction_loss)
            
        else:
            # Not an autoencoder - just compute classification loss
            output = self.base_model(base_input)
            if isinstance(output, tuple):
                window_logits_flat = output[0]
            else:
                window_logits_flat = output
            
            labeled_mask = labels_expanded >= 0
            if not labeled_mask.any():
                return None
                
            if hasattr(self.base_model, "class_weights"):
                weight = self.base_model.class_weights
            else:
                weight = None
                
            base_loss = F.cross_entropy(
                window_logits_flat[labeled_mask], 
                labels_expanded[labeled_mask], 
                weight=weight
            )
        
        # Get aggregated sample-level predictions for metrics
        sample_logits = self.forward((time_series, global_features))
        if isinstance(sample_logits, tuple):
            sample_logits = sample_logits[0]

        # Compute F1 on sample-level predictions (what matters for evaluation!)
        labeled_mask = labels >= 0
        if labeled_mask.any():
            sample_preds = sample_logits[labeled_mask].argmax(dim=-1)
            f1 = self.f1_metric(sample_preds, labels[labeled_mask])
        else:
            f1 = 0.0

        # Logging
        self.log("train_loss", base_loss, prog_bar=True)
        self.log("train_F1", f1, prog_bar=True)

        return base_loss

    def validation_step(self, batch, batch_idx):
        """Validation step that respects base model's loss computation."""
        (time_series, global_features), labels = batch
        batch_size = time_series.shape[0]

        # Create windows and expand inputs for window-level processing
        windows, num_windows = self.create_windows(time_series)
        windows_flat = windows.view(-1, windows.shape[2], windows.shape[3])
        global_features_expanded = (
            global_features.unsqueeze(1)
            .expand(-1, num_windows, -1)
            .reshape(-1, global_features.shape[1])
        )
        
        # Expand labels for each window
        labels_expanded = labels.unsqueeze(1).expand(-1, num_windows).reshape(-1)
        
        # Create batch for base model with windowed data
        base_input = (windows_flat, global_features_expanded)
        
        # For autoencoders, manually compute losses to respect base model's logic
        if hasattr(self.base_model, 'computeReconstructionLoss') and hasattr(self.base_model, 'computePredictionLoss'):
            # This is a LightningAutoencoder - compute its losses manually
            # Forward pass through base model
            predictions, (reconstructed_timeSeries, reconstructed_globalFeatures) = self.base_model(base_input)
            
            # Compute reconstruction losses
            reconstruction_loss_ts, reconstruction_loss_gf = self.base_model.computeReconstructionLoss(
                windows_flat, global_features_expanded,
                reconstructed_timeSeries, reconstructed_globalFeatures
            )
            reconstruction_loss = reconstruction_loss_ts + reconstruction_loss_gf
            
            # Compute prediction loss
            prediction_loss = self.base_model.computePredictionLoss(labels_expanded, predictions)
            
            # Combined loss using base model's weighting
            reconstruction_weight = self.base_model.reconstruction_loss_weight
            base_loss = reconstruction_weight * reconstruction_loss + (1 - reconstruction_weight) * prediction_loss
            
            # Log base model's internal metrics
            self.log("val_reconstruction_loss_timeSeries", reconstruction_loss_ts)
            self.log("val_reconstruction_loss_globalFeatures", reconstruction_loss_gf)
            self.log("val_reconstruction_loss", reconstruction_loss)
            self.log("val_prediction_loss", prediction_loss)
            
        else:
            # Not an autoencoder - just compute classification loss
            output = self.base_model(base_input)
            if isinstance(output, tuple):
                window_logits_flat = output[0]
            else:
                window_logits_flat = output
            
            labeled_mask = labels_expanded >= 0
            if not labeled_mask.any():
                return None
                
            if hasattr(self.base_model, "class_weights"):
                weight = self.base_model.class_weights
            else:
                weight = None
                
            base_loss = F.cross_entropy(
                window_logits_flat[labeled_mask], 
                labels_expanded[labeled_mask], 
                weight=weight
            )

        # Get aggregated sample-level predictions for metrics
        sample_logits = self.forward((time_series, global_features))
        if isinstance(sample_logits, tuple):
            sample_logits = sample_logits[0]

        # Compute F1 on sample-level predictions
        labeled_mask = labels >= 0
        if labeled_mask.any():
            sample_preds = sample_logits[labeled_mask].argmax(dim=-1)
            f1 = self.f1_metric(sample_preds, labels[labeled_mask])
        else:
            return None

        # Logging
        self.log("val_loss", base_loss, prog_bar=True)
        self.log("val_F1", f1, prog_bar=True)

        return base_loss

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        (time_series, global_features), _ = batch

        # Forward pass
        sample_logits = self.forward((time_series, global_features))
        if isinstance(sample_logits, tuple):
            sample_logits = sample_logits[0]

        # Get predictions and probabilities
        sample_probs = F.softmax(sample_logits, dim=-1)
        predictions = sample_probs.argmax(dim=-1)

        return predictions, sample_probs

    def configure_optimizers(self):
        """Configure optimizer for the wrapped model."""
        # Get optimizer config from base model but use wrapper's parameters
        if hasattr(self.base_model, "learning_rate"):
            lr = self.base_model.learning_rate
        else:
            lr = 0.001

        if hasattr(self.base_model, "regularization_weight"):
            weight_decay = self.base_model.regularization_weight
        else:
            weight_decay = 0.0001

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer

    def on_train_epoch_end(self):
        """Reset metrics at epoch end."""
        self.f1_metric.reset()

    def on_validation_epoch_end(self):
        """Reset metrics at epoch end."""
        self.f1_metric.reset()
