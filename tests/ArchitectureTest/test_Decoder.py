import pytest
import torch
import torch.nn as nn
from GradientGang.Pipeline.Architectures.Decoder import Decoder
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class TestDecoder:
    """Test suite for the Decoder class."""

    @pytest.fixture
    def basic_conv_transpose_params(self):
        """Basic convolutional transpose decoder parameters."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 128,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 1,
                    }
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "output_padding": 0,
                    }
                }
            ]
        }

    @pytest.fixture
    def linear_params(self):
        """Linear layer decoder parameters."""
        return {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 784,
                    }
                }
            ]
        }

    @pytest.fixture
    def mixed_params(self):
        """Mixed architecture with Linear and ConvTranspose2d layers."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 12544,
                    }
                },
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (64, 14, 14),
                    }
                },
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
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 1,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "output_padding": 0,
                    }
                }
            ]
        }

    @pytest.fixture
    def params_with_output_activation(self):
        """Decoder parameters with output activation function."""
        return {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 784,
                    }
                }
            ]
        }

    def test_decoder_initialization_basic(self, basic_conv_transpose_params):
        """Test that decoder initializes correctly with basic parameters."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )
        assert isinstance(decoder, nn.Module)
        assert hasattr(decoder, 'net')
        assert isinstance(decoder.net, nn.Sequential)

    def test_decoder_initialization_linear(self, linear_params):
        """Test decoder initialization with linear layers."""
        decoder = Decoder(
            params=linear_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )
        assert isinstance(decoder, nn.Module)
        assert isinstance(decoder.net, nn.Sequential)

    def test_decoder_initialization_mixed(self, mixed_params):
        """Test decoder initialization with mixed architecture."""
        decoder = Decoder(
            params=mixed_params,
            latent_dim=128,
            base_channel_size=32,
            num_output_channels=1
        )
        assert isinstance(decoder, nn.Module)
        assert isinstance(decoder.net, nn.Sequential)

    def test_decoder_forward_conv_transpose(self, basic_conv_transpose_params):
        """Test forward pass with convolutional transpose layers."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        # Create a dummy input tensor (batch_size=2, channels=128, height=8, width=8)
        x = torch.randn(2, 128, 8, 8)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 3  # output channels from last conv layer

    def test_decoder_forward_linear(self, linear_params):
        """Test forward pass with linear layers."""
        decoder = Decoder(
            params=linear_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Create a dummy input tensor (batch_size=4, features=128)
        x = torch.randn(4, 128)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size preserved
        assert output.shape[1] == 784  # output features from last linear layer

    def test_decoder_forward_mixed(self, mixed_params):
        """Test forward pass with mixed architecture."""
        decoder = Decoder(
            params=mixed_params,
            latent_dim=128,
            base_channel_size=32,
            num_output_channels=1
        )

        # Create a dummy input tensor
        x = torch.randn(2, 128)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 1  # output channels

    def test_decoder_activation_functions(self):
        """Test that different activation functions are correctly instantiated."""
        activation_functions = ["ReLU", "GELU", "LeakyReLU"]

        for act_fn in activation_functions:
            params = {
                "activation_function": act_fn,
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 256,
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 256,
                            "out_features": 512,
                        }
                    }
                ]
            }

            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

            # Check that the activation function is in the sequential module
            modules_list = list(decoder.net.modules())
            activation_types = {
                "ReLU": nn.ReLU,
                "GELU": nn.GELU,
                "LeakyReLU": nn.LeakyReLU
            }

            has_activation = any(isinstance(
                m, activation_types[act_fn]) for m in modules_list)
            assert has_activation, f"Activation function {act_fn} not found in decoder"

    def test_decoder_output_activation(self, params_with_output_activation):
        """Test that output activation is correctly applied to the last layer."""
        decoder = Decoder(
            params=params_with_output_activation,
            latent_dim=64,
            base_channel_size=64,
            num_output_channels=1
        )

        # Check that Sigmoid is in the modules
        modules_list = list(decoder.net.modules())
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in modules_list)
        assert has_sigmoid, "Sigmoid output activation not found in decoder"

        # Test that the output is in [0, 1] range
        x = torch.randn(2, 64)
        output = decoder.forward(x)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_decoder_no_output_activation(self, linear_params):
        """Test decoder without output activation function."""
        decoder = Decoder(
            params=linear_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Check that no sigmoid or tanh is in the modules
        modules_list = list(decoder.net.modules())
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in modules_list)
        has_tanh = any(isinstance(m, nn.Tanh) for m in modules_list)
        assert not has_sigmoid, "Sigmoid should not be in decoder without output_activation"
        assert not has_tanh, "Tanh should not be in decoder without output_activation"

    def test_decoder_output_activation_tanh(self):
        """Test decoder with Tanh output activation."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Tanh",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 784,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=64,
            base_channel_size=64,
            num_output_channels=1
        )

        # Check that Tanh is in the modules
        modules_list = list(decoder.net.modules())
        has_tanh = any(isinstance(m, nn.Tanh) for m in modules_list)
        assert has_tanh, "Tanh output activation not found in decoder"

        # Test that the output is in [-1, 1] range
        x = torch.randn(2, 64)
        output = decoder.forward(x)
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_decoder_layer_count_without_output_activation(self, basic_conv_transpose_params):
        """Test that the correct number of layers is created without output activation."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        # Each layer except the last should have an activation function after it
        # So 2 layers = 3 modules (2 conv layers + 1 activation for the first layer)
        modules_list = [m for m in decoder.net.children()]
        expected_count = len(basic_conv_transpose_params["layer_type"]) + \
            (len(basic_conv_transpose_params["layer_type"]) - 1)
        assert len(modules_list) == expected_count

    def test_decoder_layer_count_with_output_activation(self, params_with_output_activation):
        """Test that the correct number of layers is created with output activation."""
        decoder = Decoder(
            params=params_with_output_activation,
            latent_dim=64,
            base_channel_size=64,
            num_output_channels=1
        )

        # 1 layer + 1 output activation = 2 modules
        modules_list = [m for m in decoder.net.children()]
        assert len(modules_list) == 2

    def test_decoder_missing_activation_function(self):
        """Test that decoder raises error when activation_function is missing."""
        params = {
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                }
            ]
            # Missing "activation_function" key
        }

        with pytest.raises(KeyError):
            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

    def test_decoder_missing_layer_type(self):
        """Test that decoder raises error when layer_type is missing."""
        params = {
            "activation_function": "ReLU"
            # Missing "layer_type" key
        }

        with pytest.raises(KeyError):
            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

    def test_decoder_invalid_layer_name(self):
        """Test that decoder raises error for invalid layer name."""
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
            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

    def test_decoder_invalid_activation_function(self):
        """Test that decoder raises error for invalid activation function."""
        params = {
            "activation_function": "InvalidActivation",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                }
            ]
        }

        with pytest.raises(KeyError):
            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

    def test_decoder_invalid_output_activation(self):
        """Test that decoder raises error for invalid output activation function."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "InvalidOutputActivation",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                }
            ]
        }

        with pytest.raises(KeyError):
            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

    def test_decoder_conv_transpose_1d(self):
        """Test decoder with ConvTranspose1d layers."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose1d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 1,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=32,
            base_channel_size=16,
            num_output_channels=16
        )

        x = torch.randn(2, 32, 50)  # (batch, channels, sequence_length)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        assert output.shape[2] == 100  # Upsampled due to stride=2

    def test_decoder_empty_params_dict(self):
        """Test decoder with layer that has empty params dict."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (16, 7, 7),
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=784,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(2, 784)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4  # Unflattened to 4D tensor

    def test_decoder_gradient_flow(self, basic_conv_transpose_params):
        """Test that gradients flow through the decoder."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        x = torch.randn(2, 128, 8, 8, requires_grad=True)
        output = decoder.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_decoder_parameter_count(self, basic_conv_transpose_params):
        """Test that decoder has learnable parameters."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        param_count = sum(p.numel()
                          for p in decoder.parameters() if p.requires_grad)
        assert param_count > 0

    def test_decoder_eval_mode(self, basic_conv_transpose_params):
        """Test decoder in evaluation mode."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        decoder.eval()
        x = torch.randn(2, 128, 8, 8)

        with torch.no_grad():
            output1 = decoder.forward(x)
            output2 = decoder.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_decoder_train_mode(self, basic_conv_transpose_params):
        """Test decoder in training mode."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        decoder.train()
        assert decoder.training

    def test_decoder_with_bias_false(self):
        """Test decoder with bias=False parameter."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 784,
                        "bias": False,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Find the linear layer and check it has no bias
        for module in decoder.net.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None

    def test_decoder_different_padding_modes(self):
        """Test decoder with different padding modes."""
        padding_modes = ['zeros']  # ConvTranspose2d only supports 'zeros'

        for padding_mode in padding_modes:
            params = {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "ConvTranspose2d",
                        "params": {
                            "in_channels": 64,
                            "out_channels": 3,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 1,
                            "output_padding": 0,
                            "padding_mode": padding_mode,
                        }
                    }
                ]
            }

            decoder = Decoder(
                params=params,
                latent_dim=64,
                base_channel_size=64,
                num_output_channels=3
            )

            x = torch.randn(2, 64, 16, 16)
            output = decoder.forward(x)
            assert isinstance(output, torch.Tensor)

    def test_decoder_sequential_structure(self, basic_conv_transpose_params):
        """Test that decoder.net is properly structured as Sequential."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        assert isinstance(decoder.net, nn.Sequential)

        # Verify pattern: layer, activation, layer (no activation on last)
        modules_list = list(decoder.net.children())
        # First layer
        assert isinstance(modules_list[0], nn.ConvTranspose2d)
        # First activation
        assert isinstance(modules_list[1], nn.ReLU)
        # Last layer (no activation after)
        assert isinstance(modules_list[2], nn.ConvTranspose2d)

    def test_decoder_interpreter_attribute(self):
        """Test that decoder has the decoderInterpreter class attribute."""
        assert hasattr(Decoder, 'decoderInterpreter')
        assert isinstance(Decoder.decoderInterpreter, ParameterInterpreter)
        assert Decoder.decoderInterpreter.name == "DecoderInterpreter"

    def test_decoder_with_groups_parameter(self):
        """Test decoder with groups parameter for grouped convolutions."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "output_padding": 0,
                        "groups": 2,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=64,
            base_channel_size=32,
            num_output_channels=32
        )

        x = torch.randn(2, 64, 16, 16)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 32

    def test_decoder_with_dilation(self):
        """Test decoder with dilation parameter."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "output_padding": 0,
                        "dilation": 1,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=64,
            base_channel_size=64,
            num_output_channels=3
        )

        x = torch.randn(2, 64, 16, 16)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_decoder_single_activation_for_all_layers(self):
        """Test that the same activation function is used for all layers (except last)."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 512,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 512,
                        "out_features": 784,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Count activation functions - should all be GELU (except after last layer)
        gelu_count = sum(1 for m in decoder.net.modules()
                         if isinstance(m, nn.GELU))
        relu_count = sum(1 for m in decoder.net.modules()
                         if isinstance(m, nn.ReLU))
        leaky_relu_count = sum(
            1 for m in decoder.net.modules() if isinstance(m, nn.LeakyReLU))

        assert gelu_count == 2  # One after each layer except last
        assert relu_count == 0
        assert leaky_relu_count == 0

    def test_decoder_multiple_layers_same_activation(self):
        """Test that multiple different layer types use the same activation."""
        params = {
            "activation_function": "LeakyReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 3136,
                    }
                },
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (64, 7, 7),
                    }
                },
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
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                        "output_padding": 0,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=256,
            base_channel_size=32,
            num_output_channels=3
        )

        # All activations should be LeakyReLU (except after last layer)
        leaky_relu_count = sum(
            1 for m in decoder.net.modules() if isinstance(m, nn.LeakyReLU))
        relu_count = sum(1 for m in decoder.net.modules()
                         if isinstance(m, nn.ReLU))
        gelu_count = sum(1 for m in decoder.net.modules()
                         if isinstance(m, nn.GELU))

        assert leaky_relu_count == 3  # One after each layer except last
        assert relu_count == 0
        assert gelu_count == 0

    def test_decoder_upsampling_output_padding(self):
        """Test that output_padding correctly affects output size."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 0,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=64,
            base_channel_size=32,
            num_output_channels=32
        )

        x = torch.randn(2, 64, 8, 8)
        output = decoder.forward(x)

        # With stride=2, output should be doubled in spatial dimensions
        assert output.shape[2] == 16
        assert output.shape[3] == 16

    def test_decoder_unflatten_layer(self):
        """Test decoder with Unflatten layer."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 256,
                    }
                },
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (16, 4, 4),
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=16,
            num_output_channels=16
        )

        x = torch.randn(2, 128)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 16, 4, 4)

    def test_decoder_output_sigmoid_range(self):
        """Test that Sigmoid output activation produces values in [0, 1]."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 784,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(10, 128)
        output = decoder.forward(x)

        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    def test_decoder_output_tanh_range(self):
        """Test that Tanh output activation produces values in [-1, 1]."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Tanh",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 784,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(10, 128)
        output = decoder.forward(x)

        assert output.min().item() >= -1.0
        assert output.max().item() <= 1.0

    def test_decoder_encoder_compatibility(self):
        """Test that decoder output dimensions can match encoder input dimensions."""
        # Encoder params
        encoder_params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    }
                }
            ]
        }

        # Matching decoder params (reverse operation)
        decoder_params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 1,
                    }
                }
            ]
        }

        from GradientGang.Pipeline.Architectures.Encoder import Encoder

        encoder = Encoder(
            params=encoder_params,
            num_input_channels=3,
            base_channel_size=64,
            latent_dim=64
        )

        decoder = Decoder(
            params=decoder_params,
            latent_dim=64,
            base_channel_size=64,
            num_output_channels=3
        )

        # Test autoencoder pipeline
        x = torch.randn(2, 3, 32, 32)
        encoded = encoder.forward(x)
        decoded = decoder.forward(encoded)

        # Decoded output should have same spatial dimensions as input
        assert decoded.shape == x.shape

    def test_decoder_batch_size_flexibility(self, linear_params):
        """Test that decoder works with different batch sizes."""
        decoder = Decoder(
            params=linear_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 128)
            output = decoder.forward(x)
            assert output.shape[0] == batch_size
            assert output.shape[1] == 784

    def test_decoder_deterministic_in_eval_mode(self, basic_conv_transpose_params):
        """Test that decoder produces deterministic outputs in eval mode."""
        decoder = Decoder(
            params=basic_conv_transpose_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        decoder.eval()
        torch.manual_seed(42)
        x = torch.randn(2, 128, 8, 8)

        with torch.no_grad():
            output1 = decoder.forward(x)
            output2 = decoder.forward(x)
            output3 = decoder.forward(x)

        assert torch.allclose(output1, output2)
        assert torch.allclose(output2, output3)

    def test_decoder_conv2d_layer(self):
        """Test decoder with Conv2d layers (for refinement)."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=64,
            base_channel_size=32,
            num_output_channels=32
        )

        x = torch.randn(2, 64, 16, 16)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 32, 16, 16)

    def test_decoder_complex_architecture(self):
        """Test decoder with a complex multi-stage architecture."""
        params = {
            "activation_function": "GELU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 1024,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 1024,
                        "out_features": 3136,
                    }
                },
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (64, 7, 7),
                    }
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 64,
                        "out_channels": 32,
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 0,
                    }
                },
                {
                    "name": "ConvTranspose2d",
                    "params": {
                        "in_channels": 32,
                        "out_channels": 16,
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 1,
                        "output_padding": 0,
                    }
                },
                {
                    "name": "Conv2d",
                    "params": {
                        "in_channels": 16,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    }
                }
            ]
        }

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=3
        )

        x = torch.randn(2, 128)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3, 28, 28)
        # Check sigmoid range
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    # ========== LSTM/GRU/RNN Tests ==========

    @pytest.fixture
    def lstm_params(self):
        """LSTM layer decoder parameters."""
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
        """GRU layer decoder parameters."""
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
        """RNN layer decoder parameters."""
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
                        "in_features": 64,
                        "out_features": 128,
                    }
                },
                {
                    "name": "LSTM",
                    "params": {
                        "input_size": 128,
                        "hidden_size": 256,
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
                        "in_features": 256,
                        "out_features": 784,
                    }
                }
            ]
        }

    def test_decoder_lstm_initialization(self, lstm_params):
        """Test that decoder initializes correctly with LSTM layer."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )
        assert isinstance(decoder, nn.Module)
        assert hasattr(decoder, 'net')
        assert isinstance(decoder.net, nn.Sequential)

        # Check LSTM layer exists
        has_lstm = any(isinstance(m, nn.LSTM) for m in decoder.net.modules())
        assert has_lstm, "LSTM layer not found in decoder"

    def test_decoder_gru_initialization(self, gru_params):
        """Test that decoder initializes correctly with GRU layer."""
        decoder = Decoder(
            params=gru_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Check GRU layer exists
        has_gru = any(isinstance(m, nn.GRU) for m in decoder.net.modules())
        assert has_gru, "GRU layer not found in decoder"

    def test_decoder_rnn_initialization(self, rnn_params):
        """Test that decoder initializes correctly with RNN layer."""
        decoder = Decoder(
            params=rnn_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        # Check RNN layer exists
        has_rnn = any(isinstance(m, nn.RNN) for m in decoder.net.modules())
        assert has_rnn, "RNN layer not found in decoder"

    def test_decoder_rnn_nonlinearity_options(self):
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

            decoder = Decoder(
                params=params,
                latent_dim=128,
                base_channel_size=64,
                num_output_channels=1
            )

            x = torch.randn(4, 10, 64)
            output = decoder.forward(x)
            assert isinstance(output, torch.Tensor)

    def test_decoder_lstm_with_dropout(self):
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

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(4, 10, 64)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)

    def test_decoder_recurrent_gradient_flow(self, lstm_params):
        """Test that gradients flow through recurrent layers."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(4, 10, 64, requires_grad=True)
        output = decoder.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_decoder_recurrent_eval_mode(self, lstm_params):
        """Test recurrent decoder in evaluation mode."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        decoder.eval()
        x = torch.randn(4, 10, 64)

        with torch.no_grad():
            output1 = decoder.forward(x)
            output2 = decoder.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_decoder_has_params_attribute(self, lstm_params):
        """Test that decoder stores params attribute for recurrent layer detection."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        assert hasattr(decoder, 'params')
        assert decoder.params == lstm_params

    def test_decoder_recurrent_detection_logic(self, lstm_params):
        """Test that forward method correctly detects recurrent layers."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(4, 10, 64)

        # Should not raise any errors
        output = decoder.forward(x)
        assert isinstance(output, torch.Tensor)

    def test_decoder_recurrent_parameter_count(self, lstm_params):
        """Test that recurrent decoder has learnable parameters."""
        decoder = Decoder(
            params=lstm_params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        param_count = sum(p.numel()
                          for p in decoder.parameters() if p.requires_grad)
        assert param_count > 0

    def test_decoder_lstm_without_bias(self):
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

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(4, 10, 64)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)

        # Check that LSTM was created with bias=False
        for module in decoder.net.modules():
            if isinstance(module, nn.LSTM):
                assert module.bias == False

    def test_decoder_lstm_with_output_activation(self):
        """Test LSTM decoder with output activation."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
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

        decoder = Decoder(
            params=params,
            latent_dim=128,
            base_channel_size=64,
            num_output_channels=1
        )

        x = torch.randn(4, 10, 64)
        output = decoder.forward(x)

        assert isinstance(output, torch.Tensor)
        # Check sigmoid range
        assert output.min() >= 0.0
        assert output.max() <= 1.0
