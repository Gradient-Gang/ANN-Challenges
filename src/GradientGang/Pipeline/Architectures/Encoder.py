import torch.nn as nn
from ..Utils.ParameterInterpreter import ParameterInterpreter
import torch


class MultiScaleCNNBlock(nn.Module):
    """
    Multi-scale CNN block with parallel convolutional branches.
    Each branch uses a different kernel size to capture patterns at different temporal scales.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernel_sizes: list,
        use_dilation: bool = False,
        pooling_type: str = "max",
        activation_fn: nn.Module = nn.ReLU,
    ):
        """
        Args:
            in_channels: Number of input channels
            branch_channels: Number of output channels per branch
            kernel_sizes: List of kernel sizes for parallel branches (e.g., [3, 5, 7])
            use_dilation: If True, use dilation instead of larger kernels
            pooling_type: Type of pooling after concatenation ("max", "avg", or "none")
            activation_fn: Activation function to use after each conv
        """
        super().__init__()

        self.branches = nn.ModuleList()

        for i, k in enumerate(kernel_sizes):
            # Calculate padding to maintain sequence length
            if use_dilation:
                # Use dilation: kernel=3 with dilation=k//2
                dilation = max(1, k // 2)
                kernel = 3
                padding = dilation * (kernel - 1) // 2
            else:
                # Use larger kernels directly
                dilation = 1
                kernel = k
                padding = k // 2

            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(branch_channels),
                activation_fn(),
            )
            self.branches.append(branch)

        # Total output channels = branch_channels * num_branches
        self.out_channels = branch_channels * len(kernel_sizes)

        # Optional pooling after concatenation
        if pooling_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        elif pooling_type == "avg":
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)
        Returns:
            Concatenated output from all branches (batch, out_channels, seq_len)
        """
        # Run each branch in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)

        # Apply pooling if specified
        if self.pool is not None:
            out = self.pool(out)

        return out


class Encoder(nn.Module):

    # Define the ParameterInterpreter for the Encoder class
    encoderInterpreter: ParameterInterpreter = ParameterInterpreter(
        name="EncoderInterpreter",
        interpretation={
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "Conv2d": nn.Conv2d,
            "Linear": nn.Linear,
            "Conv1d": nn.Conv1d,
            "MaxPool1d": nn.MaxPool1d,
            "AvgPool1d": nn.AvgPool1d,
            "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
            "Flatten": nn.Flatten,
            "ConvTranspose2d": nn.ConvTranspose2d,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "RNN": nn.RNN,
            "MultiScaleCNN": "MultiScaleCNN",  # Special marker for multi-scale blocks
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
                {
                    "name": "MultiScaleCNN",
                    "params": {
                        "in_channels": int,
                        "branch_channels": int,
                        "kernel_sizes": list,
                        "use_dilation": bool,
                        "pooling_type": str,
                    },
                },
            ],
        },
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

    def __init__(
        self,
        params: dict,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
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

        # Get the activation function class (same for all layers)
        activation_fn_cls = self.encoderInterpreter.interpret(
            params["activation_function"]
        )

        modules = []
        # Iterate through the layer_type list
        for layer_params in params["layer_type"]:
            layer_name = layer_params["name"]

            # Special handling for MultiScaleCNN
            if layer_name == "MultiScaleCNN":
                layer_config = layer_params.get("params", {})
                # Pass the activation function to MultiScaleCNN
                layer_config["activation_fn"] = activation_fn_cls
                modules.append(MultiScaleCNNBlock(**layer_config))
                # MultiScaleCNN already includes activation internally, don't add extra
            else:
                # Standard layer handling
                layer_cls = self.encoderInterpreter.interpret(layer_name)
                modules.append(layer_cls(**layer_params.get("params", {})))
                # Add the same activation function after each layer
                modules.append(activation_fn_cls())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """Forward pass for encoder.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Encoded output tensor.
        """
        # Check if any recurrent layers (RNN, GRU, LSTM) are present
        has_recurrent = any(
            layer_params["name"] in ["LSTM", "GRU", "RNN"]
            for layer_params in self.params["layer_type"]
        )

        if has_recurrent:
            # For recurrent layers, handle the tuple output
            prev_was_recurrent = False
            for layer in self.net:
                if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                    # RNN layers return (output, hidden_state) or (output, (hidden, cell))
                    # output shape: (batch, seq_len, hidden_size) if batch_first=True
                    x = x.permute(0, 2, 1)  # Adjust dimensions if needed
                    x, _ = layer(x)
                    prev_was_recurrent = True
                elif isinstance(layer, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                    # Skip activation after recurrent layers (they output sequences)
                    if not prev_was_recurrent:
                        x = layer(x)
                    prev_was_recurrent = False
                else:
                    x = layer(x)
                    prev_was_recurrent = False

            # For time series classification, take the last timestep
            # This gives us (batch, hidden_size) suitable for feedforward layers
            if len(x.shape) == 3:  # (batch, seq_len, features)
                x = x[:, -1, :]  # Take last timestep: (batch, features)

            return x
        else:
            # Standard feedforward processing
            return self.net(x)
