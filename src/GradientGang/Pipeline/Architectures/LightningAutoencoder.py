import pytorch_lightning as L
import torch
from ..Utils.ParameterInterpreter import ParameterInterpreter
from .Encoder import Encoder
from .Decoder import Decoder
from .FeedForward import FeedForward
from torchmetrics import F1Score

LightningAutoencoderInterpreter = ParameterInterpreter(
    name="LightningAutoencoderInterpreter",
    interpretation={
        "LearningRate": float,
        "Patience": int
    },
    requiredParams={
        "EncoderParams": dict,
        "DecoderParams": dict,
        "FeedForwardParams": dict,
        "OutputDim": int
    }
)


class LightningAutoencoder(L.LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        LightningAutoencoderInterpreter.checkRequiredParams(params)

        # Store params for later use
        self.params = params

        encoder_params = params["EncoderParams"]
        decoder_params = params["DecoderParams"]
        feedforward_params = params["FeedForwardParams"]
        output_dim = params["OutputDim"]

        # Extract additional parameters if provided, with defaults
        num_input_channels = params.get("num_input_channels", 1)
        base_channel_size = params.get("base_channel_size", 64)
        latent_dim = params.get("latent_dim", 128)
        num_output_channels = params.get("num_output_channels", 1)
        act_fn = params.get("act_fn", torch.nn.GELU)

        self.encoder = Encoder(
            encoder_params, num_input_channels, base_channel_size, latent_dim, act_fn)
        self.decoder = Decoder(decoder_params, latent_dim,
                               base_channel_size, num_output_channels, act_fn)
        self.feedforward = FeedForward(feedforward_params)

        # Initialize F1Score metric as instance variable
        self.val_f1 = F1Score(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        predictions = self.feedforward(encoded)
        decoded = self.decoder(encoded)
        return predictions, decoded

    def configure_optimizers(self):
        learning_rate = self.params.get("LearningRate", 0.001)
        patience = self.params.get("Patience", 5)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=patience, min_lr=5e-5)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_F1'
                }
                }

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions, decoded = self.forward(x)

        # Compute reconstruction loss
        loss_fn_reconstruction = torch.nn.MSELoss()
        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded.view(decoded.size(0), -1)
        reconstruction_loss = loss_fn_reconstruction(decoded_flat, x_flat)

        # Compute prediction loss if labels provided
        if y is not None:
            # Define class weights - adjust these values based on your class distribution
            class_weights = torch.tensor(
                [1.0] * predictions.size(1), device=x.device)
            loss_fn_prediction = torch.nn.CrossEntropyLoss(
                weight=class_weights)
            prediction_loss = loss_fn_prediction(predictions, y)
        else:
            prediction_loss = 0

        loss = reconstruction_loss + prediction_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions, decoded = self.forward(x)

        # Compute reconstruction loss for logging
        loss_fn_reconstruction = torch.nn.MSELoss()
        x_flat = x.view(x.size(0), -1)
        decoded_flat = decoded.view(decoded.size(0), -1)
        reconstruction_loss = loss_fn_reconstruction(decoded_flat, x_flat)
        self.log("val_reconstruction_loss", reconstruction_loss)

        # Update the F1 metric with predictions and targets
        self.val_f1.update(predictions, y)
        f1_score = self.val_f1.compute()
        self.log("val_F1", f1_score)
        return f1_score

    def on_validation_epoch_end(self):
        """Reset F1 metric at the end of each validation epoch."""
        self.val_f1.reset()
