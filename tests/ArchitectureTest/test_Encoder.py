import pytest
import torch
import torch.nn as nn
from GradientGang.Pipeline.Architectures.Encoder import Encoder
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class TestEncoder:
    """Test suite for the Encoder class."""

    @pytest.fixture
    def basic_conv_params(self):
        """Basic convolutional encoder parameters."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                },
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    }
                }
            ]
        }

    @pytest.fixture
    def linear_params(self):
        """Linear layer encoder parameters."""
        return {
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
                        "out_features": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 128,
                    }
                }
            ]
        }

    @pytest.fixture
    def mixed_params(self):
        """Mixed architecture with Conv2d and Linear layers."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 1,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                },
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    }
                },
                {
                    "name": "Flatten",
                    "params": {}
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 12544,  # 64 * 14 * 14 after conv layers
                        "out_features": 128,
                    }
                }
            ]
        }

    def test_encoder_initialization_basic(self, basic_conv_params):
        """Test that encoder initializes correctly with basic parameters."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )
        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, 'net')
        assert isinstance(encoder.net, nn.Sequential)

    def test_encoder_initialization_linear(self, linear_params):
        """Test encoder initialization with linear layers."""
        encoder = Encoder(
            params=linear_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )
        assert isinstance(encoder, nn.Module)
        assert isinstance(encoder.net, nn.Sequential)

    def test_encoder_initialization_mixed(self, mixed_params):
        """Test encoder initialization with mixed architecture."""
        encoder = Encoder(
            params=mixed_params,
            num_input_channels=1,
            base_channel_size=32,
            latent_dim=128
        )
        assert isinstance(encoder, nn.Module)
        assert isinstance(encoder.net, nn.Sequential)

    def test_encoder_forward_conv(self, basic_conv_params):
        """Test forward pass with convolutional layers."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        # Create a dummy input tensor (batch_size=2, channels=3, height=32, width=32)
        x = torch.randn(2, 3, 32, 32)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 128  # output channels from last conv layer

    def test_encoder_forward_linear(self, linear_params):
        """Test forward pass with linear layers."""
        encoder = Encoder(
            params=linear_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Create a dummy input tensor (batch_size=4, features=784)
        x = torch.randn(4, 1, 28, 28)  # MNIST-like input
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size preserved
        assert output.shape[1] == 128  # output features from last linear layer

    def test_encoder_forward_mixed(self, mixed_params):
        """Test forward pass with mixed architecture."""
        encoder = Encoder(
            params=mixed_params,
            num_input_channels=1,
            base_channel_size=32,
            latent_dim=128
        )

        # Create a dummy input tensor
        x = torch.randn(2, 1, 28, 28)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 128  # output features

    def test_encoder_activation_functions(self):
        """Test that different activation functions are correctly instantiated."""
        activation_functions = ["ReLU", "GELU", "LeakyReLU"]

        for act_fn in activation_functions:
            params = {
                "activation_function": act_fn,
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 784,
                            "out_features": 256,
                        }
                    }
                ]
            }

            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=256
            )

            # Check that the activation function is in the sequential module
            modules_list = list(encoder.net.modules())
            activation_types = {
                "ReLU": nn.ReLU,
                "GELU": nn.GELU,
                "LeakyReLU": nn.LeakyReLU
            }

            has_activation = any(isinstance(
                m, activation_types[act_fn]) for m in modules_list)
            assert has_activation, f"Activation function {act_fn} not found in encoder"

    def test_encoder_layer_count(self, basic_conv_params):
        """Test that the correct number of layers is created."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        # Each layer should have an activation function after it
        # So 2 layers = 4 modules (2 conv layers + 2 activations)
        modules_list = [m for m in encoder.net.children()]
        assert len(modules_list) == len(basic_conv_params["layer_type"]) * 2

    def test_encoder_missing_activation_function(self):
        """Test that encoder raises error when activation_function is missing."""
        params = {
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                    }
                }
            ]
            # Missing "activation_function" key
        }

        with pytest.raises(KeyError):
            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=256
            )

    def test_encoder_missing_layer_type(self):
        """Test that encoder raises error when layer_type is missing."""
        params = {
            "activation_function": "ReLU"
            # Missing "layer_type" key
        }

        with pytest.raises(KeyError):
            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=256
            )

    def test_encoder_invalid_layer_name(self):
        """Test that encoder raises error for invalid layer name."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "InvalidLayer",
                    "params": {}
                }
            ]
        }

        with pytest.raises(KeyError):
            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=256
            )

    def test_encoder_invalid_activation_function(self):
        """Test that encoder raises error for invalid activation function."""
        params = {
            "activation_function": "InvalidActivation",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                    }
                }
            ]
        }

        with pytest.raises(KeyError):
            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=256
            )

    def test_encoder_conv_transpose_2d(self):
        """Test encoder with ConvTranspose2d layers."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 1,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=64,
            base_channel_size=32,
            latent_dim=128
        )

        x = torch.randn(2, 64, 8, 8)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2
        assert output.shape[1] == 32
        assert output.shape[2] == 16  # Upsampled due to stride=2

    def test_encoder_conv1d(self):
        """Test encoder with Conv1d layers."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Conv1d",
                    "params": {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=16,
            base_channel_size=32,
            latent_dim=64
        )

        x = torch.randn(2, 16, 100)  # (batch, channels, sequence_length)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2
        assert output.shape[1] == 32

    def test_encoder_empty_params_dict(self):
        """Test encoder with layer that has empty params dict."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Flatten",
                    "params": {}
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(2, 1, 28, 28)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 2  # Flattened output

    def test_encoder_gradient_flow(self, basic_conv_params):
        """Test that gradients flow through the encoder."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = encoder.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_encoder_parameter_count(self, basic_conv_params):
        """Test that encoder has learnable parameters."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        param_count = sum(p.numel()
                          for p in encoder.parameters() if p.requires_grad)
        assert param_count > 0

    def test_encoder_eval_mode(self, basic_conv_params):
        """Test encoder in evaluation mode."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        encoder.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output1 = encoder.forward(x)
            output2 = encoder.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_encoder_train_mode(self, basic_conv_params):
        """Test encoder in training mode."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        encoder.train()
        assert encoder.training

    def test_encoder_with_bias_false(self):
        """Test encoder with bias=False parameter."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                        "bias": False,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=256
        )

        # Find the linear layer and check it has no bias
        for module in encoder.net.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None

    def test_encoder_different_padding_modes(self):
        """Test encoder with different padding modes."""
        padding_modes = ['zeros', 'reflect', 'replicate', 'circular']

        for padding_mode in padding_modes:
            params = {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 3,
                            "out_channels": 64,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                            "padding_mode": padding_mode,
                        }
                    }
                ]
            }

            encoder = Encoder(
                params=params,
                num_input_channels=3,
                base_channel_size=64,
                latent_dim=128
            )

            x = torch.randn(2, 3, 32, 32)
            output = encoder.forward(x)
            assert isinstance(output, torch.Tensor)

    def test_encoder_sequential_structure(self, basic_conv_params):
        """Test that encoder.net is properly structured as Sequential."""
        encoder = Encoder(
            params=basic_conv_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        assert isinstance(encoder.net, nn.Sequential)

        # Verify alternating pattern of layer and activation
        modules_list = list(encoder.net.children())
        for i in range(0, len(modules_list), 2):
            # Even indices should be layers (Conv2d, Linear, etc.)
            assert isinstance(modules_list[i], (nn.Conv2d, nn.Linear, nn.Conv1d,
                                                nn.ConvTranspose2d, nn.Flatten))
            # Odd indices should be activation functions
            if i + 1 < len(modules_list):
                assert isinstance(
                    modules_list[i + 1], (nn.ReLU, nn.GELU, nn.LeakyReLU))

    def test_encoder_interpreter_attribute(self):
        """Test that encoder has the encoderInterpreter class attribute."""
        assert hasattr(Encoder, 'encoderInterpreter')
        assert isinstance(Encoder.encoderInterpreter, ParameterInterpreter)
        assert Encoder.encoderInterpreter.name == "EncoderInterpreter"

    def test_encoder_with_groups_parameter(self):
        """Test encoder with groups parameter for grouped convolutions."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "groups": 2,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=32,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(2, 32, 16, 16)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 64

    def test_encoder_with_dilation(self):
        """Test encoder with dilation parameter."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 2,
                        "dilation": 2,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(2, 3, 32, 32)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_encoder_single_activation_for_all_layers(self):
        """Test that the same activation function is used for all layers."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 512,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 512,
                        "out_features": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 128,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Count activation functions - should all be GELU
        gelu_count = sum(1 for m in encoder.net.modules()
                         if isinstance(m, nn.GELU))
        relu_count = sum(1 for m in encoder.net.modules()
                         if isinstance(m, nn.ReLU))
        leaky_relu_count = sum(
            1 for m in encoder.net.modules() if isinstance(m, nn.LeakyReLU))

        assert gelu_count == 3  # One after each layer
        assert relu_count == 0
        assert leaky_relu_count == 0

    def test_encoder_multiple_layers_same_activation(self):
        """Test that multiple different layer types use the same activation."""
        params = {
            "activation_function": "LeakyReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                },
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    }
                },
                {
                    "name": "Flatten",
                    "params": {}
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 3136,
                        "out_features": 256,
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=3,
            base_channel_size=32,
            latent_dim=256
        )

        # All activations should be LeakyReLU
        leaky_relu_count = sum(
            1 for m in encoder.net.modules() if isinstance(m, nn.LeakyReLU))
        relu_count = sum(1 for m in encoder.net.modules()
                         if isinstance(m, nn.ReLU))
        gelu_count = sum(1 for m in encoder.net.modules()
                         if isinstance(m, nn.GELU))

        assert leaky_relu_count == 4  # One after each layer
        assert relu_count == 0
        assert gelu_count == 0

    # ========== LSTM/GRU/RNN Tests ==========

    @pytest.fixture
    def lstm_params(self):
        """LSTM layer encoder parameters."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 2,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False
                    }
                }
            ]
        }

    @pytest.fixture
    def gru_params(self):
        """GRU layer encoder parameters."""
        return {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "GRU",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False
                    }
                }
            ]
        }

    @pytest.fixture
    def rnn_params(self):
        """RNN layer encoder parameters."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "RNN",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 1,
                        "nonlinearity": "tanh",
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False
                    }
                }
            ]
        }

    @pytest.fixture
    def mixed_lstm_linear_params(self):
        """Mixed architecture with Linear and LSTM layers."""
        return {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                    }
                },
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 256,
                        "hidden_size": 128,
                        "num_layers": 2,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.2,
                        "bidirectional": False
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 64,
                    }
                }
            ]
        }

    def test_encoder_lstm_initialization(self, lstm_params):
        """Test that encoder initializes correctly with LSTM layer."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )
        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, 'net')
        assert isinstance(encoder.net, nn.Sequential)

        # Check LSTM layer exists
        has_lstm = any(isinstance(m, nn.LSTM) for m in encoder.net.modules())
        assert has_lstm, "LSTM layer not found in encoder"

    def test_encoder_gru_initialization(self, gru_params):
        """Test that encoder initializes correctly with GRU layer."""
        encoder = Encoder(
            params=gru_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Check GRU layer exists
        has_gru = any(isinstance(m, nn.GRU) for m in encoder.net.modules())
        assert has_gru, "GRU layer not found in encoder"

    def test_encoder_rnn_initialization(self, rnn_params):
        """Test that encoder initializes correctly with RNN layer."""
        encoder = Encoder(
            params=rnn_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Check RNN layer exists
        has_rnn = any(isinstance(m, nn.RNN) for m in encoder.net.modules())
        assert has_rnn, "RNN layer not found in encoder"

    def test_encoder_lstm_forward(self, lstm_params):
        """Test forward pass with LSTM layer."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Encoder expects (batch_size, channels, seq_len) format
        # For LSTM with input_size=64, we need 64 channels
        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size preserved
        assert output.shape[1] == 128  # hidden_size (last timestep)

    def test_encoder_gru_forward(self, gru_params):
        """Test forward pass with GRU layer."""
        encoder = Encoder(
            params=gru_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Encoder expects (batch_size, channels, seq_len) format
        x = torch.randn(4, 64, 10)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4
        assert output.shape[1] == 128  # hidden_size (last timestep)

    def test_encoder_rnn_forward(self, rnn_params):
        """Test forward pass with RNN layer."""
        encoder = Encoder(
            params=rnn_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        # Encoder expects (batch_size, channels, seq_len) format
        x = torch.randn(4, 64, 10)
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4
        assert output.shape[1] == 128  # hidden_size (last timestep)

    def test_encoder_lstm_bidirectional(self):
        """Test LSTM with bidirectional=True."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": True
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
        output = encoder.forward(x)

        # Bidirectional doubles the hidden size, but we take last timestep
        assert output.shape[1] == 256  # 128 * 2 for bidirectional

    def test_encoder_gru_bidirectional(self):
        """Test GRU with bidirectional=True."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "GRU",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 1,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": True
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
        output = encoder.forward(x)

        # Bidirectional doubles the hidden size, but we take last timestep
        assert output.shape[1] == 256  # 128 * 2 for bidirectional

    def test_encoder_rnn_nonlinearity_options(self):
        """Test RNN with different nonlinearity options."""
        nonlinearities = ["tanh", "relu"]

        for nonlinearity in nonlinearities:
            params = {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "RNN",
                        "params": {
                            "input_size": 64,
                            "hidden_size": 128,
                            "num_layers": 1,
                            "nonlinearity": nonlinearity,
                            "bias": True,
                            "batch_first": True,
                            "dropout": 0.0,
                            "bidirectional": False
                        }
                    }
                ]
            }

            encoder = Encoder(
                params=params,
                num_input_channels=1,
                base_channel_size=64,
                latent_dim=128
            )

            x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
            output = encoder.forward(x)
            assert isinstance(output, torch.Tensor)

    def test_encoder_lstm_with_dropout(self):
        """Test LSTM with dropout parameter."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 2,
                        "bias": True,
                        "batch_first": True,
                        "dropout": 0.5,  # Apply dropout between LSTM layers
                        "bidirectional": False
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)

    def test_encoder_recurrent_eval_mode(self, lstm_params):
        """Test recurrent encoder in evaluation mode."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        encoder.eval()
        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10

        with torch.no_grad():
            output1 = encoder.forward(x)
            output2 = encoder.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_encoder_has_params_attribute(self, lstm_params):
        """Test that encoder stores params attribute for recurrent layer detection."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        assert hasattr(encoder, 'params')
        assert encoder.params == lstm_params

    def test_encoder_recurrent_detection_logic(self, lstm_params):
        """Test that forward method correctly detects recurrent layers."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10

        # Should not raise any errors
        output = encoder.forward(x)
        assert isinstance(output, torch.Tensor)

    def test_encoder_recurrent_parameter_count(self, lstm_params):
        """Test that recurrent encoder has learnable parameters."""
        encoder = Encoder(
            params=lstm_params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        param_count = sum(p.numel()
                          for p in encoder.parameters() if p.requires_grad)
        assert param_count > 0

    def test_encoder_lstm_without_bias(self):
        """Test LSTM with bias=False."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 64,
                        "hidden_size": 128,
                        "num_layers": 1,
                        "bias": False,  # No bias terms
                        "batch_first": True,
                        "dropout": 0.0,
                        "bidirectional": False
                    }
                }
            ]
        }

        encoder = Encoder(
            params=params,
            num_input_channels=1,
            base_channel_size=64,
            latent_dim=128
        )

        x = torch.randn(4, 64, 10)  # batch=4, channels=64, seq_len=10
        output = encoder.forward(x)

        assert isinstance(output, torch.Tensor)

        # Check that LSTM was created with bias=False
        for module in encoder.net.modules():
            if isinstance(module, nn.LSTM):
                assert module.bias == False
