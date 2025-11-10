import pytest
import torch
import pytorch_lightning as L
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


def get_basic_params():
    """Helper to get basic valid params."""
    return {
        "LearningRate": 0.001,
        "Patience": 5,
        "RegularizationWeight": 0.1,
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 1,
                        "out_features": 1
                    }
                }
            ]
        },
        "EncoderParams": {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Flatten",
                    "params": {}
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 128,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "DecoderParams": {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 784,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "FeedForwardParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 64,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 10,
                        "bias": True,
                        "device": None,
                        "dtype": None
                    }
                }
            ]
        },
        "OutputDim": 10
    }




