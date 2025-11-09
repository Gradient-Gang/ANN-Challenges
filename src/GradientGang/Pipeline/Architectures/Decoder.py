import torch.nn as nn
from ..Utils.ParameterInterpreter import ParameterInterpreter
import torch


class Decoder(nn.Module):

    decoderInterpreter: ParameterInterpreter = ParameterInterpreter(
        name="DecoderInterpreter",
        interpretation={
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "Conv2d": nn.Conv2d,
            "Linear": nn.Linear,
            "Conv1d": nn.Conv1d,
            "ConvTranspose2d": nn.ConvTranspose2d,
            "ConvTranspose1d": nn.ConvTranspose1d,
            "Unflatten": nn.Unflatten,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN,
        },
        requiredParams={
            "activation_function": ["ReLU", "GELU", "LeakyReLU"],
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": int,
                        "out_channels": int,
                        "kernel_size": int,
                        "stride": int,
                        "padding": int,
                        "dilation": int,
                        "groups": int,
                        "bias": bool,
                        "padding_mode": str,
                        "device": str,
                        "dtype": str,
                    },
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": int,
                        "out_features": int,
                        "bias": bool,
                        "device": str,
                        "dtype": str,
                    },
                },
                {
                    "name": "Conv1d",
                    "params": {
                        "in_channels": int,
                        "out_channels": int,
                        "kernel_size": int,
                        "stride": int,
                        "padding": int,
                        "dilation": int,
                        "groups": int,
                        "bias": bool,
                        "padding_mode": str,
                        "device": str,
                        "dtype": str,
                    },
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": int,
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
                        "dtype": str,
                    },
                },
                {
                    "name": "ConvTranspose1d",
                    "params": {
                        "in_channels": int,
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
                        "dtype": str,
                    },
                },
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": int,
                        "hidden_size": int,
                        "num_layers": int,
                        "bias": bool,
                        "batch_first": bool,
                        "dropout": float,
                        "bidirectional": bool,
                    },
                },
                {
                    "name": "GRU",
                    "params": {
                        "input_size": int,
                        "hidden_size": int,
                        "num_layers": int,
                        "bias": bool,
                        "batch_first": bool,
                        "dropout": float,
                        "bidirectional": bool,
                    },
                },
                {
                    "name": "RNN",
                    "params": {
                        "input_size": int,
                        "hidden_size": int,
                        "num_layers": int,
                        "nonlinearity": str,
                        "bias": bool,
                        "batch_first": bool,
                        "dropout": float,
                        "bidirectional": bool,
                    },
                },
            ],
        },
    )

    # Example params structure:
    # params = {
    #     "activation_function": "GELU",
    #     "output_activation": "Sigmoid",  # optional, for final layer
    #     "layer_type": [
    #         {
    #             "name": "Linear",
    #             "params": {
    #                 "in_features": 64,
    #                 "out_features": 128,
    #                 "bias": True,
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         },
    #         {
    #             "name": "ConvTranspose2d",
    #             "params": {
    #                 "in_channels": 64,
    #                 "out_channels": 3,
    #                 "kernel_size": 3,
    #                 "stride": 1,
    #                 "padding": 1,
    #                 "output_padding": 0,
    #                 "groups": 1,
    #                 "bias": True,
    #                 "dilation": 1,
    #                 "padding_mode": 'zeros',
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         }
    #     ]
    # }

    def __init__(
        self,
        params: dict,
        latent_dim: int,
        base_channel_size: int,
        num_output_channels: int,
        act_fn: object = nn.GELU,
    ):
        """Decoder.

        Args:
           latent_dim : Dimensionality of latent representation z
           base_channel_size : Number of channels we use in the last convolutional layers. Earlier layers might use a duplicate of it.
           num_output_channels : Number of output channels of the reconstructed image. For CIFAR, this parameter is 3
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
        self.decoderInterpreter.checkRequiredParams(params)

        # Store params for forward method
        self.params = params

        # Get the activation function class (same for all layers except possibly the last)
        activation_fn_cls = self.decoderInterpreter.interpret(
            params["activation_function"]
        )

        # Get output activation if specified (e.g., Sigmoid, Tanh for normalized outputs)
        output_activation = params.get("output_activation", None)

        modules = []
        # Iterate through the layer_type list
        for i, layer_params in enumerate(params["layer_type"]):
            # Create layer from params
            layer_cls = self.decoderInterpreter.interpret(layer_params["name"])
            modules.append(layer_cls(**layer_params.get("params", {})))

            # Add activation function after each layer except the last one
            if i < len(params["layer_type"]) - 1:
                modules.append(activation_fn_cls())
            else:
                # For the last layer, optionally add output activation
                if output_activation:
                    output_activation_cls = self.decoderInterpreter.interpret(
                        output_activation
                    )
                    modules.append(output_activation_cls())

        self.net = nn.Sequential(*modules)

    def forward(self, x, seq_len=None):
        """
        Forward pass for decoder.

        Args:
            x: Input tensor. Can be:
               - (batch, features) for standard decoding
               - (batch, seq_len, features) for sequence decoding
            seq_len: Optional sequence length for LSTM autoencoder reconstruction.
                     If provided and input is 2D, will repeat the embedding for each timestep.

        Returns:
            Decoded output tensor
        """
        # Check if any recurrent layers (RNN, GRU, LSTM) are present
        has_recurrent = any(
            layer_params["name"] in ["LSTM", "GRU", "RNN"]
            for layer_params in self.params["layer_type"]
        )

        if has_recurrent:
            # For LSTM autoencoder: expand 2D embeddings to 3D sequences
            # Input: (batch, hidden_size) → (batch, seq_len, hidden_size)
            if len(x.shape) == 2 and seq_len is not None:
                # Repeat the embedding for each timestep
                # (batch, seq_len, hidden_size)
                x = x.unsqueeze(1).repeat(1, seq_len, 1)

            # Process through layers, handling recurrent layer outputs
            prev_was_recurrent = False
            for layer in self.net:
                if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                    # RNN layers return (output, hidden_state) or (output, (hidden, cell))
                    x, _ = layer(x)
                    prev_was_recurrent = True
                elif isinstance(layer, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                    # Skip activation after recurrent layers (they output sequences)
                    if not prev_was_recurrent:
                        x = layer(x)
                    prev_was_recurrent = False
                else:
                    x = layer(x)
                    prev_was_recurrent = False

            return x
        else:
            # Standard feedforward processing
            return self.net(x)
