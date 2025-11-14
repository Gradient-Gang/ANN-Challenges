import torch
from pytorch_lightning import LightningModule


class EnsembleModel(LightningModule):
    """
    Weighted ensemble of K-fold models with performance-based voting.

    Combines multiple trained models using weighted majority voting where
    weights are proportional to individual model performance. Compatible with
    PyTorch Lightning training infrastructure and SubmissionGenerator.

    Features:
    - Performance-based weighted voting
    - Automatic normalization of weights
    - Compatible with WindowedModelWrapper
    - Device management
    - Soft voting (weighted average of probabilities)

    Example:
        >>> models = [fold_model_1, fold_model_2, fold_model_3]
        >>> weights = [0.9, 0.85, 0.88]  # F1 scores
        >>> ensemble = EnsembleModel(models, weights)
        >>> predictions = ensemble((time_series, global_features))
    """

    def __init__(self, models, weights):
        """
        Initialize weighted ensemble.

        Args:
            models (list): List of trained PyTorch models (K models for K folds)
            weights (list): List of weights (e.g., F1 scores). Auto-normalized to sum=1

        Raises:
            ValueError: If models/weights mismatch or empty
        """
        super().__init__()

        if not models:
            raise ValueError("Models list cannot be empty")

        if len(models) != len(weights):
            raise ValueError(
                f"Length mismatch: {len(models)} models but {len(weights)} weights"
            )

        self.models = torch.nn.ModuleList(models)

        # Normalize weights to sum to 1
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights_tensor / weights_tensor.sum())

    def forward(self, x):
        """
        Weighted forward pass through all models.

        Args:
            x: Tuple of (timeSeries, globalFeats) or batch dict

        Returns:
            torch.Tensor: Weighted average probabilities (batch_size, num_classes)
        """
        # Parse input
        if isinstance(x, tuple) and len(x) == 2:
            timeSeries, globalFeats = x
        else:
            timeSeries = x
            globalFeats = None

        all_probs = []

        # Collect predictions from all models
        for model in self.models:
            model.eval()

            with torch.no_grad():
                # Call model (handles both Direct and Autoencoder)
                output = model((timeSeries, globalFeats))

                # Extract logits (handle tuple from Autoencoder)
                logits = output[0] if isinstance(output, tuple) else output

                # Convert to probabilities
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)

        # Weighted average: sum(weight_i * probs_i)
        stacked_probs = torch.stack(all_probs)  # (n_models, batch, classes)
        weights_view = self.weights.view(-1, 1, 1)  # (n_models, 1, 1)
        weighted_probs = (stacked_probs * weights_view).sum(dim=0)  # (batch, classes)

        return weighted_probs

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        probs = self.forward(batch)
        return torch.argmax(probs, dim=1)

    def eval(self):
        """Set all models to eval mode."""
        for model in self.models:
            model.eval()
        return super().eval()

    def __len__(self):
        """Return number of models in ensemble."""
        return len(self.models)

    def __repr__(self):
        """String representation."""
        weights_str = ", ".join([f"{w:.3f}" for w in self.weights])
        return f"EnsembleModel(n_models={len(self)}, weights=[{weights_str}])"
