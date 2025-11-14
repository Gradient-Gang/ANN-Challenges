import torch.nn as nn
from ..Utils.ParameterInterpreter import ParameterInterpreter
import torch


class Decoder(nn.Module):

    # Define the ParameterInterpreter for the Decoder class
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
            "MaxPool1d": nn.MaxPool1d,
            "AvgPool1d": nn.AvgPool1d,
            "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
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
                    "name": "MaxPool1d",
                    "params": {
                        "kernel_size": int,
                        "stride": int,
                        "padding": int,
                        "dilation": int,
                        "return_indices": bool,
                        "ceil_mode": bool,
                    },
                },
                {
                    "name": "AvgPool1d",
                    "params": {
                        "kernel_size": int,
                        "stride": int,
                        "padding": int,
                        "ceil_mode": bool,
                        "count_include_pad": bool,
                    },
                },
                {
                    "name": "AdaptiveAvgPool1d",
                    "params": {
                        "output_size": int,
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
        """
        Decoder.

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

    def forward(self, x, seq_len=None, ground_truth=None):
        """
        Forward pass for decoder.

        Args:
            x: Input tensor. Can be:
               - (batch, features) for standard decoding
               - (batch, seq_len, features) for sequence decoding
            seq_len: Optional sequence length for LSTM autoencoder reconstruction.
                     If provided and input is 2D, will be used for sequence generation.
            ground_truth: Optional ground truth sequence for teacher forcing during training.
                         Shape: (batch, features, seq_len) - will be permuted internally.
                         Only used for RNN decoder during training.

        Returns:
            Decoded output tensor
        """
        # Check if any recurrent layers (RNN, GRU, LSTM) are present
        has_recurrent = any(
            layer_params["name"] in ["LSTM", "GRU", "RNN"]
            for layer_params in self.params["layer_type"]
        )

        if has_recurrent:
            # For LSTM autoencoder: use encoder output as initial hidden state
            # and generate sequences autoregressively
            if len(x.shape) == 2 and seq_len is not None:
                batch_size = x.size(0)

                # Find the RNN layer to get its configuration
                rnn_layer = None
                for layer in self.net:
                    if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                        rnn_layer = layer
                        break

                if rnn_layer is not None:
                    num_layers = rnn_layer.num_layers
                    hidden_size = rnn_layer.hidden_size
                    input_size = rnn_layer.input_size
                    num_directions = 2 if rnn_layer.bidirectional else 1

                    # Reshape encoder output as initial hidden state h0
                    # Encoder output: (batch, encoder_hidden * encoder_directions)
                    # Decoder needs: (num_layers * num_directions, batch, decoder_hidden)
                    #
                    # Cases to handle:
                    # 1. Bidirectional encoder → unidirectional decoder (410 → 205)
                    # 2. Multi-layer encoder → decoder expects num_layers * hidden (58 → 116 for 2 layers)
                    # 3. Different hidden sizes between encoder and decoder
                    # 4. Bidirectional decoder needs num_layers * 2 in first dimension

                    encoder_output_features = x.shape[1]
                    decoder_expected_features = (
                        num_layers * hidden_size * num_directions
                    )

                    # If encoder output doesn't match decoder expectation, we need to adapt
                    if encoder_output_features != decoder_expected_features:
                        # Case 1: Bidirectional encoder (2x features) → unidirectional decoder
                        if encoder_output_features == 2 * decoder_expected_features:
                            # Average forward and backward directions
                            x_forward = x[:, :decoder_expected_features]
                            x_backward = x[:, decoder_expected_features:]
                            x = (x_forward + x_backward) / 2

                        # Case 2: Encoder has fewer features than decoder expects (multi-layer decoder)
                        elif encoder_output_features < decoder_expected_features:
                            # Decoder has multiple layers but encoder only outputs last layer
                            # Replicate encoder output for all decoder layers
                            # Example: encoder outputs 58, decoder needs 116 (2 layers × 58)
                            if decoder_expected_features % encoder_output_features == 0:
                                # Replicate the encoder output for each decoder layer
                                num_replications = (
                                    decoder_expected_features // encoder_output_features
                                )
                                x = x.repeat(1, num_replications)
                            else:
                                # Pad with zeros to reach expected size
                                padding_size = (
                                    decoder_expected_features - encoder_output_features
                                )
                                x = torch.nn.functional.pad(
                                    x, (0, padding_size), mode="constant", value=0
                                )

                        # Case 3: Encoder has more features than decoder expects
                        else:
                            # Truncate or project to expected size
                            x = x[:, :decoder_expected_features]

                    # Reshape for RNN hidden state
                    # Shape must be: (num_layers * num_directions, batch, hidden_size)
                    h0 = x.view(num_layers * num_directions, batch_size, hidden_size)

                    # Initialize cell state for LSTM (not needed for GRU/RNN)
                    if isinstance(rnn_layer, nn.LSTM):
                        c0 = torch.zeros_like(h0)
                        hidden_state = (h0, c0)
                    else:
                        hidden_state = h0

                    # Prepare decoder input sequence
                    if self.training and ground_truth is not None:
                        # Training: Teacher forcing
                        # ground_truth shape: (batch, features, seq_len)
                        # Permute to: (batch, seq_len, features)
                        ground_truth_permuted = ground_truth.permute(0, 2, 1)

                        # Shift right: prepend start token (zeros), drop last timestep
                        # This creates: [START, x_0, x_1, ..., x_{T-2}]
                        # Target will be:      [x_0, x_1, x_2, ..., x_{T-1}]
                        start_token = torch.zeros(
                            batch_size, 1, input_size, device=x.device
                        )
                        decoder_input = torch.cat(
                            [start_token, ground_truth_permuted[:, :-1, :]], dim=1
                        )
                    else:
                        # Inference: Use zeros as input (rely on hidden state)
                        decoder_input = torch.zeros(
                            batch_size, seq_len, input_size, device=x.device
                        )

                    # Single forward pass through RNN (no Python loop!)
                    x, _ = rnn_layer(decoder_input, hidden_state)

                    # Continue with remaining layers (activations, projections, etc.)
                    processed_rnn = False
                    for layer in self.net:
                        if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                            if not processed_rnn:
                                processed_rnn = True
                                continue  # Skip - already processed above
                        elif isinstance(
                            layer, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)
                        ):
                            # Apply activation
                            x = layer(x)
                        else:
                            # Apply other layers (Linear, etc.)
                            x = layer(x)

                    # Permute from (batch, seq_len, features) to (batch, features, seq_len)
                    # to match the original input shape
                    if len(x.shape) == 3:
                        x = x.permute(0, 2, 1)

                    return x
                else:
                    # Fallback: no RNN layer found (shouldn't happen)
                    x = x.unsqueeze(1).repeat(1, seq_len, 1)

            # Process through layers, handling recurrent layer outputs
            prev_was_recurrent = False
            for layer in self.net:
                if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                    # RNN layers return (output, hidden_state) or (output, (hidden, cell))
                    x, _ = layer(x)
                    prev_was_recurrent = True
                elif isinstance(
                    layer, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)
                ):
                    # Skip activation after recurrent layers (they output sequences)
                    if not prev_was_recurrent:
                        x = layer(x)
                    prev_was_recurrent = False
                else:
                    x = layer(x)
                    prev_was_recurrent = False

            # Permute from (batch, seq_len, features) to (batch, features, seq_len)
            # to match the original input shape
            if len(x.shape) == 3:
                x = x.permute(0, 2, 1)

            return x
        else:
            # Standard feedforward processing
            return self.net(x)
