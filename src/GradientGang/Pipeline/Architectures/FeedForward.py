import torch.nn as nn
from ..Utils.ParameterInterpreter import ParameterInterpreter


class FeedForward(nn.Module):

    feedforwardInterpreter: ParameterInterpreter = ParameterInterpreter(
        name="FeedForwardInterpreter",
        interpretation={
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "LeakyReLU": nn.LeakyReLU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "ELU": nn.ELU,
            "SELU": nn.SELU,
            "Linear": nn.Linear,
            "Dropout": nn.Dropout,
            "BatchNorm1d": nn.BatchNorm1d,
            "LayerNorm": nn.LayerNorm,
        },
        requiredParams={
            "activation_function": ["ReLU", "GELU", "LeakyReLU", "Sigmoid", "Tanh", "ELU", "SELU"],
            "layer_type": [{"name": "Linear", "params": {"in_features": int,
                                                         "out_features": int,
                                                         "bias": bool}},
                           {"name": "Dropout", "params": {"p": float,
                                                          "inplace": bool}},
                           {"name": "BatchNorm1d", "params": {"num_features": int,
                                                              "eps": float,
                                                              "momentum": float,
                                                              "affine": bool,
                                                              "track_running_stats": bool}},
                           {"name": "LayerNorm", "params": {"normalized_shape": int,
                                                            "eps": float,
                                                            "elementwise_affine": bool}}]
        }
    )

    # Example params structure:
    # params = {
    #     "activation_function": "ReLU",
    #     "output_activation": "Sigmoid",  # optional, for final layer
    #     "layer_type": [
    #         {
    #             "name": "Linear",
    #             "params": {
    #                 "in_features": 784,
    #                 "out_features": 256,
    #                 "bias": True,
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         },
    #         {
    #             "name": "Dropout",
    #             "params": {
    #                 "p": 0.2,
    #                 "inplace": False
    #             }
    #         },
    #         {
    #             "name": "Linear",
    #             "params": {
    #                 "in_features": 256,
    #                 "out_features": 128,
    #                 "bias": True,
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         },
    #         {
    #             "name": "Linear",
    #             "params": {
    #                 "in_features": 128,
    #                 "out_features": 10,
    #                 "bias": True,
    #                 "device": None,
    #                 "dtype": None
    #             }
    #         }
    #     ]
    # }

    def __init__(self, params: dict):
        """FeedForward Network.

        Args:
           params : Dictionary containing network configuration with activation_function,
                    optional output_activation, and layer_type list defining the network architecture

        """
        super().__init__()
        self.feedforwardInterpreter.checkRequiredParams(params)

        # Get the activation function class (same for all layers except possibly the last)
        activation_fn_cls = self.feedforwardInterpreter.interpret(
            params["activation_function"]
        )

        # Get output activation if specified (e.g., Sigmoid, Tanh for normalized outputs)
        output_activation = params.get("output_activation", None)

        modules = []
        # Iterate through the layer_type list
        for i, layer_params in enumerate(params["layer_type"]):
            # Create layer from params
            layer_cls = self.feedforwardInterpreter.interpret(
                layer_params["name"])
            modules.append(layer_cls(**layer_params.get("params", {})))

            # Add activation function after each layer except the last one
            # Skip activation for non-linear layers like Dropout, BatchNorm, LayerNorm
            if i < len(params["layer_type"]) - 1 and layer_params["name"] in ["Linear"]:
                modules.append(activation_fn_cls())
            elif i == len(params["layer_type"]) - 1:
                # For the last layer, optionally add output activation
                if output_activation:
                    output_activation_cls = self.feedforwardInterpreter.interpret(
                        output_activation)
                    modules.append(output_activation_cls())

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)
