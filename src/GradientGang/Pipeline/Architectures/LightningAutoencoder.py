import pytorch_lightning as L
import torch
import yaml
from typing import Optional, Tuple
from ..Utils.ParameterInterpreter import ParameterInterpreter
from .Encoder import Encoder
from .Decoder import Decoder
from .FeedForward import FeedForward
from torchmetrics import F1Score


class LightningAutoencoder(L.LightningModule):

    # Define the ParameterInterpreter for the LightningAutoencoder class
    LightningAutoencoderInterpreter = ParameterInterpreter(
        name="LightningAutoencoderInterpreter",
        interpretation={"ClassWeightsPath": str},
        requiredParams={
            "EncoderParams": dict,
            "GlobalFFEncoderParams": dict,
            "DecoderParams": dict,
            "GlobalFFDecoderParams": dict,
            "FeedForwardParams": dict,
            "OutputDim": int,
            "LearningRate": float,
            "Patience": int,
            "RegularizationWeight": float,
            "ReconstructionLossWeight": float,
        },
    )

    def __init__(self, params: dict):
        """
        Lightning Autoencoder Architecture for combined time series and global feature data.
        Combines an encoder-decoder for time series data and a feedforward network for global features.
        The outputs are concatenated and passed through another feedforward network to produce final predictions.
        Args:
            params (dict): Configuration parameters for the architecture.
        """

        # Initialize the LightningModule and check required parameters
        super().__init__()
        self.LightningAutoencoderInterpreter.checkRequiredParams(params)

        # Store params for later use
        self.params = params

        encoder_params = params["EncoderParams"]
        global_ff_encoder_params = params["GlobalFFEncoderParams"]
        decoder_params = params["DecoderParams"]
        global_ff_decoder_params = params["GlobalFFDecoderParams"]
        feedforward_params = params["FeedForwardParams"]
        output_dim = params["OutputDim"]

        self.reconstruction_loss_weight = params.get("ReconstructionLossWeight", 0.5)
        assert (
            0 <= self.reconstruction_loss_weight <= 1.0
        ), "ReconstructionLossWeight must be between 0 and 1."

        # Load class weights from YAML file if provided
        class_weights_path = params.get("ClassWeightsPath")

        if class_weights_path:
            try:
                with open(class_weights_path, "r") as f:
                    class_weights_dict = yaml.safe_load(f)
                # Convert dict to tensor ordered by class indices (0, 1, 2, ...)
                # Assumes class labels are integers 0 to output_dim-1
                # Handle both integer and string keys in the YAML file
                class_weights_list = []
                for i in range(output_dim):
                    class_weights_list.append(class_weights_dict[i])
                class_weights_tensor = torch.tensor(
                    class_weights_list, dtype=torch.float32
                )
                # Register as buffer so it moves with the model to the correct device
                self.register_buffer("class_weights", class_weights_tensor)
            except Exception as e:
                print(
                    f"Error: Could not load class weights from {class_weights_path}. Error: {e}"
                )
        else:
            # No path provided, use equal weights (all ones)
            class_weights_tensor = torch.ones(output_dim, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights_tensor)

        # Extract additional parameters if provided, with defaults
        num_input_channels = params.get("num_input_channels", 1)
        base_channel_size = params.get("base_channel_size", 64)
        latent_dim = params.get("latent_dim", 128)
        num_output_channels = params.get("num_output_channels", 1)
        act_fn = params.get("act_fn", torch.nn.GELU)

        # Initialize the Encoder, Decoder, and FeedForward networks
        self.encoder = Encoder(
            encoder_params, num_input_channels, base_channel_size, latent_dim, act_fn
        )
        self.decoder = Decoder(
            decoder_params, latent_dim, base_channel_size, num_output_channels, act_fn
        )
        self.globalff_encoder = FeedForward(global_ff_encoder_params)
        self.globalff_decoder = FeedForward(global_ff_decoder_params)

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

        # Initialize F1Score metric as instance variable
        self.val_f1 = F1Score(task="multiclass", num_classes=output_dim)

        # Initialize loss functions
        self.reconstructionLossFunction = torch.nn.MSELoss()

        weight_tensor: Optional[torch.Tensor] = (
            self.class_weights if isinstance(self.class_weights, torch.Tensor) else None
        )
        self.predictionLossFunction = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, x) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Lightning Autoencoder architecture.
        Args:
            x (tuple): A tuple containing time series data and global features.
        Returns:
            tuple: Predictions and a tuple of decoded time series and decoded global features.
        """

        # Store original sequence length for decoder reconstruction
        timeSeries = x[0]
        globalFeatures = x[1]
        # timeSeries shape: (batch, features, seq_len) = (batch, 34, 160)
        # We need seq_len which is shape[2]
        original_seq_len = timeSeries.shape[2] if len(timeSeries.shape) == 3 else None

        encoded_timeSeries = self.encoder(timeSeries)
        encoded_globalFeatures = self.globalff_encoder(globalFeatures)
        combined_encoded = torch.cat(
            (encoded_timeSeries, encoded_globalFeatures), dim=1
        )
        predictions = self.feedforward(combined_encoded)
        # Pass sequence length to decoder for LSTM autoencoder reconstruction
        decoded = self.decoder(encoded_timeSeries, seq_len=original_seq_len)
        decoded_globalFeatures = self.globalff_decoder(encoded_globalFeatures)

        return predictions, (decoded, decoded_globalFeatures)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Returns:
            dict: Dictionary containing optimizer and scheduler configurations.
        """

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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def computeReconstructionLoss(
        self,
        timeSeriesTrue: torch.Tensor,
        globalFeaturesTrue: torch.Tensor,
        timeSeriesReconstructed: torch.Tensor,
        globalFeaturesReconstructed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes both the time series and global reconstruction loss

        Args:
            timeSeriesTrue (torch.Tensor): The target time series data
            globalFeaturesTrue (torch.Tensor): The target global features
            timeSeriesReconstructed (torch.Tensor): The reconstructed time series
            globalFeaturesReconstructed (torch.Tensor): The reconstructed global features

        Returns:
            _type_: _description_
        """
        # Time Series
        timeSeries_flat = timeSeriesTrue.reshape(timeSeriesTrue.size(0), -1)
        timeSeriesReconstructed_flat = timeSeriesReconstructed.reshape(
            timeSeriesReconstructed.size(0), -1
        )
        reconstruction_loss_timeSeries: torch.Tensor = self.reconstructionLossFunction(
            timeSeriesReconstructed_flat,
            timeSeries_flat,
        )

        # Global Features
        globalFeatures_flat = globalFeaturesTrue.reshape(globalFeaturesTrue.size(0), -1)
        globalFeaturesReconstructed_flat = globalFeaturesReconstructed.reshape(
            globalFeaturesReconstructed.size(0), -1
        )
        reconstruction_loss_globalFeatures: torch.Tensor = (
            self.reconstructionLossFunction(
                globalFeaturesReconstructed_flat, globalFeatures_flat
            )
        )

        return reconstruction_loss_timeSeries, reconstruction_loss_globalFeatures

    def computePredictionLoss(
        self, classTargets: torch.Tensor, classPredictions: torch.Tensor
    ) -> torch.Tensor:
        """Computes the prediction loss

        Args:
            classTargets (torch.Tensor): the class targets
            classPredictions (torch.Tensor): the class predictions

        Returns:
            torch.Tensor: Th
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
        """Computes the F1 score

        Args:
            classTargets (torch.Tensor): the class targets
            classPredictions (torch.Tensor): the class predictions

        Returns:
            float: Computed F1 score
        """

        labeled_mask = classTargets >= 0
        f1 = 0.0
        if labeled_mask.any():
            availableTargets = classTargets[labeled_mask]
            availablePredictionsLogits = classPredictions[labeled_mask]
            availablePredictions = torch.argmax(availablePredictionsLogits, dim=1)
            f1 = self.val_f1(availablePredictions, availableTargets)
        return f1

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
        classPredictions, (reconstructedTimeSeries, reconstructedGlobalFeatures) = (
            self.forward(inputData)
        )

        # Compute reconstruction loss
        reconstructionLossTimeSeries, reconstructionLossGlobalFeatures = (
            self.computeReconstructionLoss(
                timeSeries,
                globalFeatures,
                reconstructedTimeSeries,
                reconstructedGlobalFeatures,
            )
        )
        reconstructionLoss = (
            reconstructionLossTimeSeries + reconstructionLossGlobalFeatures
        )

        # Compute prediction loss
        predictionLoss = self.computePredictionLoss(classTargets, classPredictions)

        # Compute complete loss
        loss = (
            self.reconstruction_loss_weight * reconstructionLoss
            + (1 - self.reconstruction_loss_weight) * predictionLoss
        )

        # Compute F1 score
        f1 = self.computeF1Score(classTargets, classPredictions)

        # Logging
        self.log("train_reconstruction_loss_timeSeries", reconstructionLossTimeSeries)
        self.log(
            "train_reconstruction_loss_globalFeatures", reconstructionLossGlobalFeatures
        )
        self.log("train_reconstruction_loss", reconstructionLoss)
        self.log("train_prediction_loss", predictionLoss)
        self.log("train_loss", loss)
        self.log("train_F1", f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for Lightning Autoencoder architecture.
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
        classPredictions, (reconstructedTimeSeries, reconstructedGlobalFeatures) = (
            self.forward(inputData)
        )

        # Compute reconstruction loss
        reconstructionLossTimeSeries, reconstructionLossGlobalFeatures = (
            self.computeReconstructionLoss(
                timeSeries,
                globalFeatures,
                reconstructedTimeSeries,
                reconstructedGlobalFeatures,
            )
        )
        reconstructionLoss = (
            reconstructionLossTimeSeries + reconstructionLossGlobalFeatures
        )

        # Compute prediction loss
        predictionLoss = self.computePredictionLoss(classTargets, classPredictions)

        # Compute complete loss
        loss = (
            self.reconstruction_loss_weight * reconstructionLoss
            + (1 - self.reconstruction_loss_weight) * predictionLoss
        )

        # f1 F1 score
        f1 = self.computeF1Score(classTargets, classPredictions)

        # Logging
        self.log("val_reconstruction_loss_timeSeries", reconstructionLossTimeSeries)
        self.log(
            "val_reconstruction_loss_globalFeatures", reconstructionLossGlobalFeatures
        )
        self.log("val_reconstruction_loss", reconstructionLoss)
        self.log("val_prediction_loss", predictionLoss)
        self.log("val_loss", loss)
        self.log("val_F1", f1, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Reset F1 metric at the end of each validation epoch.
        """
        self.val_f1.reset()

    def on_train_epoch_end(self):
        """
        Reset F1 metric at the end of each training epoch.
        """
        self.val_f1.reset()

    def get_embeddings(self, x):
        """
        Get latent embeddings from the encoder using no_grad.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Latent embeddings from the encoder.
        """
        with torch.no_grad():
            embeddings = self.encoder(x)
        return embeddings
