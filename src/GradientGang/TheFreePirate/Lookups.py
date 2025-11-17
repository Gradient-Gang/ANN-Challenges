import torch


activationFunctionsLookup = {
    "relu": torch.nn.ReLU,
    "leakyrelu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
    "selu": torch.nn.SELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
}
