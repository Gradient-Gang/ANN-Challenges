import pytorch_lightning as L
import torch
import yaml
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
            "Validation": bool,
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

        self.validation = params.get("Validation", True)

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
        feedforward_params["layer_type"].append(
            {
                "name": "Linear",
                "params": {
                    "in_features": feedforward_params["layer_type"][-1]["params"][
                        "out_features"
                    ],
                    "out_features": output_dim - 1,
                    "bias": True,
                },
            }
        )
        self.feedforward = FeedForward(feedforward_params)
        # Load class weights from YAML file if provided

        class_weights_path = params.get("ClassWeightsPath")

        if class_weights_path:
            try:
                with open(class_weights_path, 'r') as f:
                    class_weights_dict = yaml.safe_load(f)
                # Convert dict to tensor ordered by class indices (0, 1, 2, ...)
                # Assumes class labels are integers 0 to output_dim-1
                # Handle both integer and string keys in the YAML file
                class_weights_list = []
                for i in range(output_dim):
                    class_weights_list.append(class_weights_dict[i])
                class_weights_tensor = torch.tensor(
                    class_weights_list, dtype=torch.float32)
                # Register as buffer so it moves with the model to the correct device
                self.register_buffer('class_weights', class_weights_tensor)
            except Exception as e:
                print(
                    f"Error: Could not load class weights from {class_weights_path}. Error: {e}")
        else:
            # No path provided, use equal weights (all ones)
            class_weights_tensor = torch.ones(output_dim, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights_tensor)

        # Initialize F1Score metric as instance variable
        self.val_f1 = F1Score(task="multiclass", num_classes=output_dim)

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

        # Append a column of zeros to the predictions
        zero_tensor = torch.zeros(
            (predictions.size(0), 1), device=predictions.device)
        predictions = torch.cat((predictions, zero_tensor), dim=-1)

        # Return final predictions
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
        monitor = "val_loss" if self.validation else "train_loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": monitor},
        }

    def training_step(self, batch, batch_idx):
        """
        Training step for Direct architecture.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: Computed loss for the batch.
        """

        # Unpack batch
        x, y = batch
        predictions = self.forward(x)

        if y is not None:
            # Define class weights - adjust these values based on your class distribution
            loss_fn_prediction = torch.nn.CrossEntropyLoss(
                weight=self.class_weights)
            loss = loss_fn_prediction(predictions, y)
        else:
            loss = 0

        self.log("train_loss", loss)
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
        x, y = batch
        predictions = self.forward(x)
        if y is not None:
            # Define class weights - adjust these values based on your class distribution
            loss_fn_prediction = torch.nn.CrossEntropyLoss(
                weight=self.class_weights)
            loss = loss_fn_prediction(predictions, y)
        else:
            loss = 0
        # Update the F1 metric with predictions and targets
        self.val_f1.update(predictions, y)
        f1_score = self.val_f1.compute()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_F1", f1_score, prog_bar=True)
        return f1_score

    def on_validation_epoch_end(self):
        """
        Reset F1 metric at the end of each validation epoch.
        """
        self.val_f1.reset()
