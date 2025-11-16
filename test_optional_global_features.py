"""
Quick test to verify optional global features implementation
"""
import torch
from src.GradientGang.Pipeline.Architectures.Direct import Direct
from src.GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder


def test_direct_without_global():
    """Test Direct architecture without global features"""
    print("Testing Direct without global features...")

    params = {
        "EncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 34,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                    }
                }
            ]
        },
        # GlobalFFEncoderParams is None (not provided)
        "FeedForwardParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {"in_features": 64, "out_features": 32, "bias": True}
                }
            ]
        },
        "OutputDim": 3,
        "LearningRate": 0.001,
        "RegularizationWeight": 0.0,
    }

    model = Direct(params)
    print(f"  use_global_features: {model.use_global_features}")

    # Test forward pass
    batch_size = 2
    time_series = torch.randn(batch_size, 34, 160)
    global_features = torch.randn(batch_size, 10)  # Still provided but ignored

    output = model((time_series, global_features))
    print(f"  Output shape: {output.shape}")
    assert output.shape == (
        batch_size, 3), f"Expected shape (2, 3), got {output.shape}"
    print("  ✓ Direct without global features works!")


def test_direct_with_global():
    """Test Direct architecture with global features"""
    print("\nTesting Direct with global features...")

    params = {
        "EncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 34,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                    }
                }
            ]
        },
        "GlobalFFEncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {"in_features": 10, "out_features": 20, "bias": True}
                }
            ]
        },
        "FeedForwardParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    # 64 + 20 = 84
                    "params": {"in_features": 84, "out_features": 32, "bias": True}
                }
            ]
        },
        "OutputDim": 3,
        "LearningRate": 0.001,
        "RegularizationWeight": 0.0,
    }

    model = Direct(params)
    print(f"  use_global_features: {model.use_global_features}")

    # Test forward pass
    batch_size = 2
    time_series = torch.randn(batch_size, 34, 160)
    global_features = torch.randn(batch_size, 10)

    output = model((time_series, global_features))
    print(f"  Output shape: {output.shape}")
    assert output.shape == (
        batch_size, 3), f"Expected shape (2, 3), got {output.shape}"
    print("  ✓ Direct with global features works!")


def test_autoencoder_without_global():
    """Test LightningAutoencoder without global features"""
    print("\nTesting LightningAutoencoder without global features...")

    params = {
        "EncoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 34,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                    }
                }
            ]
        },
        "DecoderParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 34,
                        "hidden_size": 64,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {"in_features": 64, "out_features": 34, "bias": True}
                }
            ]
        },
        # GlobalFFEncoderParams and GlobalFFDecoderParams are None
        "FeedForwardParams": {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {"in_features": 64, "out_features": 32, "bias": True}
                }
            ]
        },
        "OutputDim": 3,
        "LearningRate": 0.001,
        "RegularizationWeight": 0.0,
        "ReconstructionLossWeight": 0.5,
    }

    model = LightningAutoencoder(params)
    print(f"  use_global_features: {model.use_global_features}")

    # Test forward pass
    batch_size = 2
    time_series = torch.randn(batch_size, 34, 160)
    global_features = torch.randn(batch_size, 10)  # Still provided but ignored

    predictions, (decoded_ts, decoded_gf) = model(
        (time_series, global_features))
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Decoded time series shape: {decoded_ts.shape}")
    print(f"  Decoded global features: {decoded_gf}")

    assert predictions.shape == (
        batch_size, 3), f"Expected predictions shape (2, 3), got {predictions.shape}"
    assert decoded_ts.shape == (
        batch_size, 34, 160), f"Expected decoded_ts shape (2, 34, 160), got {decoded_ts.shape}"
    assert decoded_gf is None, f"Expected decoded_gf to be None, got {type(decoded_gf)}"
    print("  ✓ LightningAutoencoder without global features works!")


if __name__ == "__main__":
    test_direct_without_global()
    test_direct_with_global()
    test_autoencoder_without_global()
    print("\n✅ All tests passed!")
