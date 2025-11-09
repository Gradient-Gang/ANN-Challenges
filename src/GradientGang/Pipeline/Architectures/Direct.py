import pytorch_lightning as L
import torch
from ..Utils.ParameterInterpreter import ParameterInterpreter
from .Encoder import Encoder
from .FeedForward import FeedForward
from torchmetrics import F1Score


class Direct(L.LightningModule):

    DirectInterpreter = ParameterInterpreter(
        name="DirectInterpreter",
        interpretation={
            "LearningRate": float,
            "Patience": int
        },
        requiredParams={
            "EncoderParams": dict,
            "GlobalFFEncoderParams": dict,
            "FeedForwardParams": dict,
            "OutputDim": int
        }
    )

    def __init__(self, params: dict):
        super().__init__()
        self.DirectInterpreter.checkRequiredParams(params)

        # Store params for later use
        self.params = params

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
            encoder_params, num_input_channels, base_channel_size, latent_dim, act_fn)
        self.globalff_encoder = FeedForward(global_ff_encoder_params)
        feedforward_params["layer_type"].append({
            "name": "Linear",
            "params": {
                "in_features": feedforward_params["layer_type"][-1]["params"]["out_features"],
                "out_features": output_dim-1,
                "bias": True,
            }
        })
        self.feedforward = FeedForward(feedforward_params)

        # Initialize F1Score metric as instance variable
        self.val_f1 = F1Score(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        timeSeries = x[0]
        globalFeatures = x[1]
        encoded_timeSeries = self.encoder(timeSeries)
        encoded_globalFeatures = self.globalff_encoder(globalFeatures)
        combined_encoded = torch.cat(
            (encoded_timeSeries, encoded_globalFeatures), dim=1)
        predictions = self.feedforward(combined_encoded)
        zero_tensor = torch.zeros(
            (predictions.size(0), 1), device=predictions.device)
        predictions = torch.cat((predictions, zero_tensor), dim=-1)
        return predictions

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
        predictions = self.forward(x)

        if y is not None:
            # Define class weights - adjust these values based on your class distribution
            class_weights = torch.tensor(
                [1.0] * predictions.size(1), device=x.device)
            loss_fn_prediction = torch.nn.CrossEntropyLoss(
                weight=class_weights)
            loss = loss_fn_prediction(predictions, y)
        else:
            loss = 0

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x)

        # Update the F1 metric with predictions and targets
        self.val_f1.update(predictions, y)
        f1_score = self.val_f1.compute()
        self.log("val_F1", f1_score)
        return f1_score

    def on_validation_epoch_end(self):
        """Reset F1 metric at the end of each validation epoch."""
        self.val_f1.reset()
