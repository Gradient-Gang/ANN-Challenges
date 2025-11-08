import pytest
import torch
import torch.nn as nn
from GradientGang.Pipeline.Architectures.FeedForward import FeedForward
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class TestFeedForward:
    """Test suite for the FeedForward class."""

    @pytest.fixture
    def basic_linear_params(self):
        """Basic feedforward network parameters."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 128,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    @pytest.fixture
    def params_with_dropout(self):
        """Feedforward network with dropout layers."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 512,
                        "bias": True,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.5,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 512,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.3,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    @pytest.fixture
    def params_with_batch_norm(self):
        """Feedforward network with batch normalization."""
        return {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "BatchNorm1d",
                    "params": {
                        "num_features": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 128,
                        "bias": True,
                    }
                },
                {
                    "name": "BatchNorm1d",
                    "params": {
                        "num_features": 128,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    @pytest.fixture
    def params_with_layer_norm(self):
        """Feedforward network with layer normalization."""
        return {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 512,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "LayerNorm",
                    "params": {
                        "normalized_shape": 256,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 128,
                        "bias": True,
                    }
                },
                {
                    "name": "LayerNorm",
                    "params": {
                        "normalized_shape": 128,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    @pytest.fixture
    def params_with_output_activation(self):
        """Feedforward network with output activation."""
        return {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    @pytest.fixture
    def complex_params(self):
        """Complex feedforward network with mixed layers."""
        return {
            "activation_function": "GELU",
            "output_activation": "Tanh",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,
                        "out_features": 512,
                        "bias": True,
                    }
                },
                {
                    "name": "BatchNorm1d",
                    "params": {
                        "num_features": 512,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.4,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 512,
                        "out_features": 256,
                        "bias": True,
                    }
                },
                {
                    "name": "LayerNorm",
                    "params": {
                        "normalized_shape": 256,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.3,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 256,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

    def test_feedforward_initialization_basic(self, basic_linear_params):
        """Test that feedforward initializes correctly with basic parameters."""
        model = FeedForward(params=basic_linear_params)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'network')
        assert isinstance(model.network, nn.Sequential)

    def test_feedforward_initialization_with_dropout(self, params_with_dropout):
        """Test feedforward initialization with dropout layers."""
        model = FeedForward(params=params_with_dropout)
        assert isinstance(model, nn.Module)
        assert isinstance(model.network, nn.Sequential)

    def test_feedforward_initialization_with_batch_norm(self, params_with_batch_norm):
        """Test feedforward initialization with batch normalization."""
        model = FeedForward(params=params_with_batch_norm)
        assert isinstance(model, nn.Module)
        assert isinstance(model.network, nn.Sequential)

    def test_feedforward_initialization_with_layer_norm(self, params_with_layer_norm):
        """Test feedforward initialization with layer normalization."""
        model = FeedForward(params=params_with_layer_norm)
        assert isinstance(model, nn.Module)
        assert isinstance(model.network, nn.Sequential)

    def test_feedforward_initialization_complex(self, complex_params):
        """Test feedforward initialization with complex architecture."""
        model = FeedForward(params=complex_params)
        assert isinstance(model, nn.Module)
        assert isinstance(model.network, nn.Sequential)

    def test_feedforward_forward_basic(self, basic_linear_params):
        """Test forward pass with basic linear layers."""
        model = FeedForward(params=basic_linear_params)

        # Create a dummy input tensor (batch_size=4, features=784)
        x = torch.randn(4, 784)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size preserved
        assert output.shape[1] == 10  # output features

    def test_feedforward_forward_with_dropout(self, params_with_dropout):
        """Test forward pass with dropout layers."""
        model = FeedForward(params=params_with_dropout)

        x = torch.randn(8, 784)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (8, 10)

    def test_feedforward_forward_with_batch_norm(self, params_with_batch_norm):
        """Test forward pass with batch normalization."""
        model = FeedForward(params=params_with_batch_norm)

        # Batch norm needs batch_size > 1
        x = torch.randn(16, 784)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (16, 10)

    def test_feedforward_forward_with_layer_norm(self, params_with_layer_norm):
        """Test forward pass with layer normalization."""
        model = FeedForward(params=params_with_layer_norm)

        x = torch.randn(8, 512)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (8, 10)

    def test_feedforward_forward_complex(self, complex_params):
        """Test forward pass with complex architecture."""
        model = FeedForward(params=complex_params)

        x = torch.randn(16, 784)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (16, 10)

    def test_feedforward_activation_functions(self):
        """Test that different activation functions are correctly instantiated."""
        activation_functions = ["ReLU", "GELU", "LeakyReLU", "ELU", "SELU"]

        for act_fn in activation_functions:
            params = {
                "activation_function": act_fn,
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 100,
                            "out_features": 50,
                            "bias": True,
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 50,
                            "out_features": 10,
                            "bias": True,
                        }
                    }
                ]
            }

            model = FeedForward(params=params)

            # Check that the activation function is in the sequential module
            modules_list = list(model.network.modules())
            activation_types = {
                "ReLU": nn.ReLU,
                "GELU": nn.GELU,
                "LeakyReLU": nn.LeakyReLU,
                "ELU": nn.ELU,
                "SELU": nn.SELU,
            }

            has_activation = any(isinstance(
                m, activation_types[act_fn]) for m in modules_list)
            assert has_activation, f"Activation function {act_fn} not found in model"

    def test_feedforward_output_activation(self, params_with_output_activation):
        """Test that output activation is correctly applied to the last layer."""
        model = FeedForward(params=params_with_output_activation)

        # Check that Sigmoid is in the modules
        modules_list = list(model.network.modules())
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in modules_list)
        assert has_sigmoid, "Sigmoid output activation not found in model"

        # Test that the output is in [0, 1] range
        x = torch.randn(4, 784)
        output = model.forward(x)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_feedforward_no_output_activation(self, basic_linear_params):
        """Test feedforward without output activation function."""
        model = FeedForward(params=basic_linear_params)

        # Check that no sigmoid or tanh is in the modules
        modules_list = list(model.network.modules())
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in modules_list)
        has_tanh = any(isinstance(m, nn.Tanh) for m in modules_list)
        assert not has_sigmoid, "Sigmoid should not be in model without output_activation"
        assert not has_tanh, "Tanh should not be in model without output_activation"

    def test_feedforward_output_activation_tanh(self):
        """Test feedforward with Tanh output activation."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Tanh",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 50,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Check that Tanh is in the modules
        modules_list = list(model.network.modules())
        has_tanh = any(isinstance(m, nn.Tanh) for m in modules_list)
        assert has_tanh, "Tanh output activation not found in model"

        # Test that the output is in [-1, 1] range
        x = torch.randn(4, 100)
        output = model.forward(x)
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_feedforward_missing_activation_function(self):
        """Test that feedforward raises error when activation_function is missing."""
        params = {
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                }
            ]
            # Missing "activation_function" key
        }

        with pytest.raises(KeyError):
            model = FeedForward(params=params)

    def test_feedforward_missing_layer_type(self):
        """Test that feedforward raises error when layer_type is missing."""
        params = {
            "activation_function": "ReLU"
            # Missing "layer_type" key
        }

        with pytest.raises(KeyError):
            model = FeedForward(params=params)

    def test_feedforward_invalid_layer_name(self):
        """Test that feedforward raises error for invalid layer name."""
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
            model = FeedForward(params=params)

    def test_feedforward_invalid_activation_function(self):
        """Test that feedforward raises error for invalid activation function."""
        params = {
            "activation_function": "InvalidActivation",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                }
            ]
        }

        with pytest.raises(KeyError):
            model = FeedForward(params=params)

    def test_feedforward_invalid_output_activation(self):
        """Test that feedforward raises error for invalid output activation."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "InvalidOutputActivation",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                }
            ]
        }

        with pytest.raises(KeyError):
            model = FeedForward(params=params)

    def test_feedforward_gradient_flow(self, basic_linear_params):
        """Test that gradients flow through the feedforward network."""
        model = FeedForward(params=basic_linear_params)

        x = torch.randn(4, 784, requires_grad=True)
        output = model.forward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_feedforward_parameter_count(self, basic_linear_params):
        """Test that feedforward has learnable parameters."""
        model = FeedForward(params=basic_linear_params)

        param_count = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)
        assert param_count > 0

    def test_feedforward_eval_mode(self, basic_linear_params):
        """Test feedforward in evaluation mode."""
        model = FeedForward(params=basic_linear_params)

        model.eval()
        x = torch.randn(4, 784)

        with torch.no_grad():
            output1 = model.forward(x)
            output2 = model.forward(x)

        # In eval mode with same input, output should be identical
        assert torch.allclose(output1, output2)

    def test_feedforward_train_mode(self, basic_linear_params):
        """Test feedforward in training mode."""
        model = FeedForward(params=basic_linear_params)

        model.train()
        assert model.training

    def test_feedforward_dropout_train_vs_eval(self, params_with_dropout):
        """Test that dropout behaves differently in train vs eval mode."""
        model = FeedForward(params=params_with_dropout)
        x = torch.randn(100, 784)

        # Training mode - dropout should be active
        model.train()
        with torch.no_grad():
            output_train1 = model.forward(x)
            output_train2 = model.forward(x)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode - dropout should be inactive
        model.eval()
        with torch.no_grad():
            output_eval1 = model.forward(x)
            output_eval2 = model.forward(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)

    def test_feedforward_batch_norm_train_vs_eval(self, params_with_batch_norm):
        """Test that batch normalization behaves differently in train vs eval mode."""
        model = FeedForward(params=params_with_batch_norm)
        x = torch.randn(16, 784)

        # Training mode
        model.train()
        output_train = model.forward(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model.forward(x)

        # Outputs will be different due to batch norm behavior
        assert not torch.allclose(output_train, output_eval)

    def test_feedforward_with_bias_false(self):
        """Test feedforward with bias=False parameter."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 50,
                        "out_features": 10,
                        "bias": False,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Check that all linear layers have no bias
        linear_count = 0
        for module in model.network.modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None
                linear_count += 1

        assert linear_count == 2

    def test_feedforward_interpreter_attribute(self):
        """Test that feedforward has the feedforwardInterpreter class attribute."""
        assert hasattr(FeedForward, 'feedforwardInterpreter')
        assert isinstance(FeedForward.feedforwardInterpreter,
                          ParameterInterpreter)
        assert FeedForward.feedforwardInterpreter.name == "FeedForwardInterpreter"

    def test_feedforward_sequential_structure(self, basic_linear_params):
        """Test that feedforward.network is properly structured as Sequential."""
        model = FeedForward(params=basic_linear_params)

        assert isinstance(model.network, nn.Sequential)

        # Verify structure: layer, activation, layer, activation, layer (no activation on last)
        modules_list = list(model.network.children())
        # 3 layers + 2 activations = 5 modules
        assert len(modules_list) == 5

        # Check pattern
        assert isinstance(modules_list[0], nn.Linear)
        assert isinstance(modules_list[1], nn.ReLU)
        assert isinstance(modules_list[2], nn.Linear)
        assert isinstance(modules_list[3], nn.ReLU)
        assert isinstance(modules_list[4], nn.Linear)

    def test_feedforward_single_layer(self):
        """Test feedforward with a single layer."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        x = torch.randn(4, 100)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 10)

        # Should have only the linear layer (no activation after last layer)
        modules_list = list(model.network.children())
        assert len(modules_list) == 1

    def test_feedforward_deep_network(self):
        """Test feedforward with many layers."""
        layer_sizes = [784, 512, 384, 256, 128, 64, 32, 10]
        layer_type = []

        for i in range(len(layer_sizes) - 1):
            layer_type.append({
                "name": "Linear",
                "params": {
                    "in_features": layer_sizes[i],
                    "out_features": layer_sizes[i + 1],
                    "bias": True,
                }
            })

        params = {
            "activation_function": "ReLU",
            "layer_type": layer_type
        }

        model = FeedForward(params=params)

        x = torch.randn(4, 784)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 10)

    def test_feedforward_batch_size_flexibility(self, basic_linear_params):
        """Test that feedforward works with different batch sizes."""
        model = FeedForward(params=basic_linear_params)

        batch_sizes = [1, 4, 8, 16, 32]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 784)
            output = model.forward(x)
            assert output.shape[0] == batch_size
            assert output.shape[1] == 10

    def test_feedforward_deterministic_in_eval_mode(self, basic_linear_params):
        """Test that feedforward produces deterministic outputs in eval mode."""
        model = FeedForward(params=basic_linear_params)

        model.eval()
        torch.manual_seed(42)
        x = torch.randn(4, 784)

        with torch.no_grad():
            output1 = model.forward(x)
            output2 = model.forward(x)
            output3 = model.forward(x)

        assert torch.allclose(output1, output2)
        assert torch.allclose(output2, output3)

    def test_feedforward_activation_after_linear_only(self, params_with_dropout):
        """Test that activation is only added after Linear layers, not after Dropout."""
        model = FeedForward(params=params_with_dropout)

        modules_list = list(model.network.children())

        # Check that ReLU only appears after Linear layers
        for i, module in enumerate(modules_list):
            if isinstance(module, nn.ReLU):
                # Previous module should be Linear
                assert isinstance(modules_list[i - 1], nn.Linear)

    def test_feedforward_no_activation_after_normalization(self, params_with_batch_norm):
        """Test that activation is not added after normalization layers."""
        model = FeedForward(params=params_with_batch_norm)

        modules_list = list(model.network.children())

        # Check that BatchNorm1d is never followed by activation
        for i, module in enumerate(modules_list):
            if isinstance(module, nn.BatchNorm1d) and i + 1 < len(modules_list):
                next_module = modules_list[i + 1]
                # Next module should not be an activation (should be Linear)
                assert not isinstance(next_module, (nn.ReLU, nn.GELU, nn.LeakyReLU,
                                                    nn.Sigmoid, nn.Tanh, nn.ELU, nn.SELU))

    def test_feedforward_mixed_layers_structure(self, complex_params):
        """Test the structure of a complex network with mixed layers."""
        model = FeedForward(params=complex_params)

        modules_list = list(model.network.children())

        # Verify expected layer types in sequence
        # Activation is added AFTER Linear layers only (and before next layer)
        expected_types = [
            nn.Linear,      # 0 - first Linear
            nn.GELU,        # 1 - activation after first Linear
            nn.BatchNorm1d,  # 2 - BatchNorm (no activation after)
            nn.Dropout,     # 3 - Dropout (no activation after)
            nn.Linear,      # 4 - second Linear
            nn.GELU,        # 5 - activation after second Linear
            nn.LayerNorm,   # 6 - LayerNorm (no activation after)
            nn.Dropout,     # 7 - Dropout (no activation after)
            nn.Linear,      # 8 - third Linear (last layer)
            nn.Tanh,        # 9 - output activation
        ]

        assert len(modules_list) == len(expected_types)

        for i, (module, expected_type) in enumerate(zip(modules_list, expected_types)):
            assert isinstance(
                module, expected_type), f"Module {i} should be {expected_type}, got {type(module)}"

    def test_feedforward_output_sigmoid_range(self):
        """Test that Sigmoid output activation produces values in [0, 1]."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        x = torch.randn(50, 100)
        output = model.forward(x)

        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    def test_feedforward_output_tanh_range(self):
        """Test that Tanh output activation produces values in [-1, 1]."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Tanh",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        x = torch.randn(50, 100)
        output = model.forward(x)

        assert output.min().item() >= -1.0
        assert output.max().item() <= 1.0

    def test_feedforward_layer_norm_single_batch(self, params_with_layer_norm):
        """Test that layer norm works with single batch (unlike batch norm)."""
        model = FeedForward(params=params_with_layer_norm)

        # Layer norm should work with batch_size=1
        x = torch.randn(1, 512)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 10)

    def test_feedforward_dropout_probability(self):
        """Test different dropout probabilities."""
        dropout_probs = [0.1, 0.3, 0.5, 0.7]

        for p in dropout_probs:
            params = {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 100,
                            "out_features": 50,
                            "bias": True,
                        }
                    },
                    {
                        "name": "Dropout",
                        "params": {
                            "p": p,
                            "inplace": False,
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 50,
                            "out_features": 10,
                            "bias": True,
                        }
                    }
                ]
            }

            model = FeedForward(params=params)
            x = torch.randn(16, 100)
            output = model.forward(x)

            assert isinstance(output, torch.Tensor)
            assert output.shape == (16, 10)

    def test_feedforward_elu_activation(self):
        """Test feedforward with ELU activation."""
        params = {
            "activation_function": "ELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 50,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Check that ELU is in the modules
        modules_list = list(model.network.modules())
        has_elu = any(isinstance(m, nn.ELU) for m in modules_list)
        assert has_elu, "ELU activation not found in model"

        x = torch.randn(4, 100)
        output = model.forward(x)
        assert isinstance(output, torch.Tensor)

    def test_feedforward_selu_activation(self):
        """Test feedforward with SELU activation."""
        params = {
            "activation_function": "SELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 50,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 50,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Check that SELU is in the modules
        modules_list = list(model.network.modules())
        has_selu = any(isinstance(m, nn.SELU) for m in modules_list)
        assert has_selu, "SELU activation not found in model"

        x = torch.randn(4, 100)
        output = model.forward(x)
        assert isinstance(output, torch.Tensor)

    def test_feedforward_single_activation_for_all_layers(self):
        """Test that the same activation function is used for all layers (except last)."""
        params = {
            "activation_function": "GELU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 100,
                        "out_features": 80,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 80,
                        "out_features": 60,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 60,
                        "out_features": 10,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Count activation functions - should all be GELU (except after last layer)
        gelu_count = sum(1 for m in model.network.modules()
                         if isinstance(m, nn.GELU))
        relu_count = sum(1 for m in model.network.modules()
                         if isinstance(m, nn.ReLU))
        leaky_relu_count = sum(
            1 for m in model.network.modules() if isinstance(m, nn.LeakyReLU))

        assert gelu_count == 2  # One after each layer except last
        assert relu_count == 0
        assert leaky_relu_count == 0

    def test_feedforward_parameter_initialization(self, basic_linear_params):
        """Test that parameters are properly initialized."""
        model = FeedForward(params=basic_linear_params)

        # Check that weights are not all zeros
        for param in model.parameters():
            if param.requires_grad:
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_feedforward_empty_params_dict(self):
        """Test feedforward behavior with minimal params."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 10,
                        "out_features": 5,
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)
        x = torch.randn(2, 10)
        output = model.forward(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 5)

    def test_feedforward_classification_task(self):
        """Test feedforward for a classification task (MNIST-like)."""
        params = {
            "activation_function": "ReLU",
            "output_activation": "Sigmoid",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 784,  # 28x28 flattened
                        "out_features": 128,
                        "bias": True,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.2,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 128,
                        "out_features": 64,
                        "bias": True,
                    }
                },
                {
                    "name": "Dropout",
                    "params": {
                        "p": 0.2,
                        "inplace": False,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 10,  # 10 classes
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        # Test with batch of flattened MNIST images
        x = torch.randn(32, 784)
        output = model.forward(x)

        assert output.shape == (32, 10)
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    def test_feedforward_regression_task(self):
        """Test feedforward for a regression task."""
        params = {
            "activation_function": "ReLU",
            "layer_type": [
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 20,
                        "out_features": 64,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 64,
                        "out_features": 32,
                        "bias": True,
                    }
                },
                {
                    "name": "Linear",
                    "params": {
                        "in_features": 32,
                        "out_features": 1,  # Single output for regression
                        "bias": True,
                    }
                }
            ]
        }

        model = FeedForward(params=params)

        x = torch.randn(16, 20)
        output = model.forward(x)

        assert output.shape == (16, 1)

    def test_feedforward_optimizer_compatibility(self, basic_linear_params):
        """Test that feedforward is compatible with PyTorch optimizers."""
        model = FeedForward(params=basic_linear_params)

        # Test with different optimizers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(4, 784)
        target = torch.randn(4, 10)

        output = model.forward(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Check that optimization step executed successfully
        assert True

    def test_feedforward_loss_backward(self, basic_linear_params):
        """Test that loss backward propagation works correctly."""
        model = FeedForward(params=basic_linear_params)

        x = torch.randn(4, 784)
        target = torch.randn(4, 10)

        output = model.forward(x)
        loss = nn.MSELoss()(output, target)

        # Get initial parameter values
        initial_params = [p.clone() for p in model.parameters()]

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_feedforward_state_dict(self, basic_linear_params):
        """Test that model state can be saved and loaded."""
        model1 = FeedForward(params=basic_linear_params)
        model2 = FeedForward(params=basic_linear_params)

        # Get state dict from model1
        state_dict = model1.state_dict()

        # Load into model2
        model2.load_state_dict(state_dict)

        # Test that outputs are identical
        x = torch.randn(4, 784)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1.forward(x)
            output2 = model2.forward(x)

        assert torch.allclose(output1, output2)
