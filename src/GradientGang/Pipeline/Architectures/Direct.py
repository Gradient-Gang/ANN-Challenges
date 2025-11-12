import pytorch_lightning as L
import torch
import yaml
from typing import Optional, Literal
from ..Utils.ParameterInterpreter import ParameterInterpreter
from .Encoder import Encoder
from .FeedForward import FeedForward
from torchmetrics import F1Score


class Direct(L.LightningModule):

    # Define the ParameterInterpreter for the Direct class
    DirectInterpreter = ParameterInterpreter(
        name="DirectInterpreter",
        interpretation={
            "ClassWeightsPath": str,
            "SchedulerMonitoringTarget": str,  # TODO: find a way to enforce this to be either "val" or "train"
        },
        requiredParams={
            "EncoderParams": dict,
            "GlobalFFEncoderParams": dict,
            "FeedForwardParams": dict,
            "OutputDim": int,
            "LearningRate": float,
            "Patience": int,
            "RegularizationWeight": float,
        },
    )

    def __init__(self, params: dict):
        """
        Direct Architecture for classification tasks.
        Combines an encoder for time series data and a feedforward network for global features.
        The outputs are concatenated and passed through another feedforward network to produce final predictions.

        Args:
            params (dict): Configuration parameters for the architecture.
        """

        # Initialize the LightningModule and check required parameters
        super().__init__()
        self.DirectInterpreter.checkRequiredParams(params)

        # Store params for later use
        self.params = params

        self.schedulerMonitoringTarget = params.get("SchedulerMonitoringTarget", "val")

        encoder_params = params["EncoderParams"]
        global_ff_encoder_params = params["GlobalFFEncoderParams"]
        feedforward_params = params["FeedForwardParams"]
        output_dim = params["OutputDim"]

        # Extract additional parameters if provided, with defaults
        num_input_channels = params.get("num_input_channels", 1)
        base_channel_size = params.get("base_channel_size", 64)
        latent_dim = params.get("latent_dim", 128)
        act_fn = params.get("act_fn", torch.nn.GELU)

        self.encoder = Encoder(
            encoder_params, num_input_channels, base_channel_size, latent_dim, act_fn
        )
        self.globalff_encoder = FeedForward(global_ff_encoder_params)

        # Add a final linear layer to feedforward to match output_dim
        feedforward_params["layer_type"].append(
            {
                "name": "Linear",
                "params": {
                    "in_features": feedforward_params["layer_type"][-1]["params"][
                        "out_features"
                    ],
                    "out_features": output_dim,
                    "bias": True,
                },
            }
        )
        self.feedforward = FeedForward(feedforward_params)
        # Load class weights from YAML file if provided

        class_weights_path = params.get("ClassWeightsPath")

        # Default: ones
        class_weights_tensor = torch.ones(output_dim, dtype=torch.float32)

        if class_weights_path:
            try:
                with open(class_weights_path, "r") as f:
                    loaded = yaml.safe_load(f)

                # YAML can load a list or a dict. Normalize to list of length output_dim.
                if isinstance(loaded, dict):
                    # Support string keys '0', '1' as well as int keys
                    class_weights_list = []
                    for i in range(output_dim):
                        # Try int key, then str key
                        if i in loaded:
                            class_weights_list.append(loaded[i])
                        elif str(i) in loaded:
                            class_weights_list.append(loaded[str(i)])
                        else:
                            raise KeyError(f"Missing class weight for index {i}")
                    class_weights_tensor = torch.tensor(
                        class_weights_list, dtype=torch.float32
                    )
                elif isinstance(loaded, (list, tuple)):
                    if len(loaded) != output_dim:
                        raise ValueError(
                            f"Class weights length {len(loaded)} does not match OutputDim {output_dim}"
                        )
                    class_weights_tensor = torch.tensor(
                        list(loaded), dtype=torch.float32
                    )
                else:
                    raise TypeError("ClassWeights YAML must contain a dict or list")

            except Exception as e:
                # Fall back to ones but print a concise warning
                print(
                    f"Warning: could not load class weights from {class_weights_path}: {e}; using ones"
                )

        # Register as buffer so it moves with the model to the correct device
        self.register_buffer("class_weights", class_weights_tensor)

        # Initialize F1Score metric as instance variable
        self.f1Function = F1Score(task="multiclass", num_classes=output_dim)

        # Prepare prediction loss function using the registered class_weights buffer
        weight_tensor: Optional[torch.Tensor] = (
            self.class_weights if isinstance(self.class_weights, torch.Tensor) else None
        )
        self.predictionLossFunction = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    def computePredictionLoss(
        self, classTargets: torch.Tensor, classPredictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction loss while ignoring unlabeled targets (labels < 0).

        Returns a tensor on the module device.
        """
        labeled_mask = classTargets >= 0
        if labeled_mask.any():
            availableTargets = classTargets[labeled_mask]
            availablePredictionsLogits = classPredictions[labeled_mask]
            prediction_loss = self.predictionLossFunction(
                availablePredictionsLogits, availableTargets
            )
        else:
            prediction_loss = torch.tensor(0.0, device=self.device)

        return prediction_loss

    def computeF1Score(
        self, classTargets: torch.Tensor, classPredictions: torch.Tensor
    ) -> float:
        """Compute F1 score for available (non-negative) targets.

        Uses internal `self.val_f1` metric; does not reset it.
        """
        labeled_mask = classTargets >= 0
        f1 = 0.0
        if labeled_mask.any():
            availableTargets = classTargets[labeled_mask]
            availablePredictionsLogits = classPredictions[labeled_mask]
            availablePredictions = torch.argmax(availablePredictionsLogits, dim=1)
            f1 = self.f1Function(availablePredictions, availableTargets)
        return f1

    def forward(self, x):
        """
        Forward pass for Direct architecture.
        Args:
            x (tuple): A tuple containing time series data and global features.
        Returns:
            torch.Tensor: Predictions with an additional zero column.

        """
        # Unpack input tuple
        timeSeries = x[0]
        globalFeatures = x[1]
        encoded_timeSeries = self.encoder(timeSeries)
        encoded_globalFeatures = self.globalff_encoder(globalFeatures)

        # Combine encoded features and pass through feedforward network
        combined_encoded = torch.cat(
            (encoded_timeSeries, encoded_globalFeatures), dim=1
        )
        predictions = self.feedforward(combined_encoded)

        # Return final predictions (shape: batch x output_dim)
        return predictions

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Returns:
            dict: Dictionary containing optimizer and scheduler configurations.
        """

        # Extract optimizer parameters from self.params
        learning_rate = self.params.get("LearningRate", 0.001)
        patience = self.params.get("Patience", 5)
        regularization_weight = self.params.get("RegularizationWeight", 0.0)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=regularization_weight
        )
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=patience, min_lr=5e-5
        )
        monitor = f"{self.schedulerMonitoringTarget}_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": monitor},
        }

    def training_step(self, batch, batch_idx):
        """
        Training step for Lightning Autoencoder architecture.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        # Decode inputs
        inputData, classTargets = batch
        timeSeries, globalFeatures = inputData

        # Compute predictions
        classPredictions = self.forward(inputData)

        # Compute prediction loss
        predictionLoss = self.computePredictionLoss(classTargets, classPredictions)

        # Compute complete loss
        loss = predictionLoss

        # Compute F1 score
        f1 = self.computeF1Score(classTargets, classPredictions)

        # Logging
        self.log("train_prediction_loss", predictionLoss)
        self.log("train_loss", loss)
        self.log("train_F1", f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for Direct architecture.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        Returns:
            float: Computed F1 score for the batch.
        """
        # Decode inputs
        inputData, classTargets = batch
        timeSeries, globalFeatures = inputData

        # Compute predictions
        classPredictions = self.forward(inputData)

        # Compute prediction loss
        predictionLoss = self.computePredictionLoss(classTargets, classPredictions)

        # Compute complete loss
        loss = predictionLoss

        # Compute F1 score
        f1 = self.computeF1Score(classTargets, classPredictions)

        # Logging
        self.log("val_prediction_loss", predictionLoss)
        self.log("val_loss", loss)
        self.log("val_F1", f1, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Reset F1 metric at the end of each validation epoch.
        """
        self.f1Function.reset()

    def on_train_epoch_end(self):
        """
        Reset F1 metric at the end of each training epoch.
        """
        self.f1Function.reset()
