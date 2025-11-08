import torch.nn as nn
from ..Utils.ParameterInterpreter import ParameterInterpreter
import torch


class Encoder(nn.Module):

    encoderInterpreter: ParameterInterpreter = ParameterInterpreter(
        name="EncoderInterpreter",
        interpretation={
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "Conv2d": nn.Conv2d,
            "Linear": nn.Linear,
            "Conv1d": nn.Conv1d,
            "Flatten": nn.Flatten,
            "ConvTranspose2d": nn.ConvTranspose2d,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN,

        },    requiredParams={
            "activation_function": ["ReLU", "GELU", "LeakyReLU"],
            "layer_type": [{"name": "Conv2d", "params": {"in_channels": int,
                                                         "out_channels": int,
                                                         "kernel_size": int,
                                                         "stride": int,
                                                         "padding": int,
                                                         "dilation": int,
                                                         "groups": int,
                                                         "bias": bool,
                                                         "padding_mode": str,
                                                         "device": str,
                                                         "dtype": str}},
                           {"name": "Linear", "params": {"in_features": int,
                                                         "out_features": int,
                                                         "bias": bool,
                                                         "device": str,
                                                         "dtype": str}},
                           {"name": "Conv1d", "params": {"in_channels": int,
                                                         "out_channels": int,
                                                         "kernel_size": int,
                                                         "stride": int,
                                                         "padding": int,
                                                         "dilation": int,
                                                         "groups": int,
                                                         "bias": bool,
                                                         "padding_mode": str,
                                                         "device": str,
                                                         "dtype": str}},
                           {"name": "ConvTranspose2d", "params": {"in_channels": int,
                                                                  "out_channels": int,
                                                                  "kernel_size": int,
                                                                  "stride": int,
                                                                  "padding": int,
                                                                  "output_padding": int,
                                                                  "groups": int,
                                                                  "bias": bool,
                                                                  "dilation": int,
                                                                  "padding_mode": str,
                                                                  "device": str,
                                                                  "dtype": str}},
                           {"name": "LSTM", "params": {"input_size": int,
                                                       "hidden_size": int,
                                                       "num_layers": int,
                                                       "bias": bool,
                                                       "batch_first": bool,
                                                       "dropout": float,
                                                       "bidirectional": bool}},
                           {"name": "GRU", "params": {"input_size": int,
                                                      "hidden_size": int,
                                                      "num_layers": int,
                                                      "bias": bool,
                                                      "batch_first": bool,
                                                      "dropout": float,
                                                      "bidirectional": bool}},
                           {"name": "RNN", "params": {"input_size": int,
                                                      "hidden_size": int,
                                                      "num_layers": int,
                                                      "nonlinearity": str,
                                                      "bias": bool,
                                                      "batch_first": bool,
                                                      "dropout": float,
                                                      "bidirectional": bool}},
                           ]

        }
    )

    # params = {
    #     "activation_function": "GELU",
    #     "layer_type": [
    #         {
    #             "name": "Conv2d",
    #             "params": {
    #                 "in_channels": 3,
    #                 "out_channels": 64,
    #                 "kernel_size": 3,
    #                 "stride": 1,
    #                 "padding": 1,
    #                 "dilation": 1,
    #                 "groups": 1,
    #                 "bias": True,
    #                 "padding_mode": 'zeros',
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         },
    #         {
    #             "name": "Linear",
    #             "params": {
    #                 "in_features": 128,
    #                 "out_features": 64,
    #                 "bias": True,
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         }
    #     ]
    # }

    def __init__(self, params: dict, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()
        self.encoderInterpreter.checkRequiredParams(params)

        # Store params for forward method
        self.params = params

        c_hid = base_channel_size

        # Get the activation function class (same for all layers)
        activation_fn_cls = self.encoderInterpreter.interpret(
            params["activation_function"]
        )

        modules = []
        # Iterate through the layer_type list
        for layer_params in params["layer_type"]:
            # create layer from params
            layer_cls = self.encoderInterpreter.interpret(layer_params["name"])
            modules.append(layer_cls(**layer_params.get("params", {})))
            # Add the same activation function after each layer
            modules.append(activation_fn_cls())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        # Check if any recurrent layers (RNN, GRU, LSTM) are present
        has_recurrent = any(
            layer_params["name"] in ["LSTM", "GRU", "RNN"]
            for layer_params in self.params["layer_type"]
        )

        if has_recurrent:
            # For recurrent layers, handle the tuple output
            for layer in self.net:
                if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                    # RNN layers return (output, hidden_state) or (output, (hidden, cell))
                    # We take the output and continue
                    x, _ = layer(x)
                else:
                    x = layer(x)
            return x
        else:
            # Standard feedforward processing
            return self.net(x)
