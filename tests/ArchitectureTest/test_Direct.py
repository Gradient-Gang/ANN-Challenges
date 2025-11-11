import pytest
import torch
import torch.nn as nn
import pytorch_lightning as L
from GradientGang.Pipeline.Architectures.Direct import Direct
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class TestDirect:
    """Test suite for the Direct class."""

    @pytest.fixture
    def basic_params(self):
        """Basic parameters for Direct model."""
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
                        "name": "Flatten",
                        "params": {}
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 25088,  # 32 * 28 * 28
                            "out_features": 128,
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
                            "in_features": 129,  # 128 from encoder + 1 from global features
                            "out_features": 64,
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 64,
                            "out_features": 10,
                        }
                    }
                ]
            },
            "OutputDim": 10
        }

    @pytest.fixture
    def params_with_optional(self):
        """Parameters with optional settings."""
        return {
            "LearningRate": 0.0005,
            "Patience": 3,
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
                        "name": "Conv2d",
                        "params": {
                            "in_channels": 3,
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
                            "in_features": 12544,  # 64 * 14 * 14
                            "out_features": 256,
                        }
                    }
                ]
            },
            "FeedForwardParams": {
                "activation_function": "GELU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 256,
                            "out_features": 128,
                        }
                    },
                    {
                        "name": "Dropout",
                        "params": {
                            "p": 0.2,
                            "inplace": False
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 128,
                            "out_features": 5,
                        }
                    }
                ]
            },
            "OutputDim": 5,
            "num_input_channels": 3,
            "base_channel_size": 64,
            "latent_dim": 256,
            "act_fn": torch.nn.GELU
        }

    def test_direct_initialization_basic(self, basic_params):
        """Test that Direct model initializes correctly with basic parameters."""
        model = Direct(basic_params)
        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'feedforward')
        assert hasattr(model, 'val_f1')

    def test_direct_initialization_with_optional(self, params_with_optional):
        """Test Direct initialization with optional parameters."""
        model = Direct(params_with_optional)
        assert isinstance(model, L.LightningModule)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'feedforward')

    def test_direct_missing_required_params(self):
        """Test that Direct raises error when required parameters are missing."""
        # Missing EncoderParams
        params = {
            "LearningRate": 0.001,
            "Patience": 5,
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing FeedForwardParams
        params = {
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
            "EncoderParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing OutputDim
        params = {
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
            "EncoderParams": {},
            "FeedForwardParams": {}
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing LearningRate
        params = {
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
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

        # Missing Patience
        params = {
            "LearningRate": 0.001,
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": 10
        }
        with pytest.raises(KeyError):
            Direct(params)

    def test_direct_configure_optimizers(self, basic_params):
        """Test that optimizer is configured correctly."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()

        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config
        assert isinstance(optimizer_config['optimizer'], torch.optim.AdamW)
        assert isinstance(optimizer_config['lr_scheduler']['scheduler'],
                          torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert optimizer_config['lr_scheduler']['monitor'] == 'val_F1'

    def test_direct_optimizer_learning_rate(self, basic_params):
        """Test that optimizer uses correct learning rate."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']

        # Check that learning rate matches the parameter
        assert optimizer.param_groups[0]['lr'] == basic_params['LearningRate']

    def test_direct_scheduler_patience(self, basic_params):
        """Test that scheduler uses correct patience."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.patience == basic_params['Patience']

    def test_direct_parameter_count(self, basic_params):
        """Test that model has learnable parameters."""
        model = Direct(basic_params)
        param_count = sum(p.numel()
                          for p in model.parameters() if p.requires_grad)
        assert param_count > 0

    def test_direct_train_mode(self, basic_params):
        """Test model in training mode."""
        model = Direct(basic_params)
        model.train()
        assert model.training

    def test_direct_different_learning_rates(self):
        """Test Direct with different learning rates."""
        learning_rates = [0.0001, 0.001, 0.01, 0.1]

        for lr in learning_rates:
            params = {
                "LearningRate": lr,
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
                    "activation_function": "ReLU",
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
                                "out_features": 10,
                            }
                        }
                    ]
                },
                "OutputDim": 10
            }

            model = Direct(params)
            optimizer_config = model.configure_optimizers()
            assert optimizer_config['optimizer'].param_groups[0]['lr'] == lr

    def test_direct_interpreter_attribute(self):
        """Test that Direct has the DirectInterpreter attribute."""
        assert hasattr(Direct, 'DirectInterpreter')
        assert isinstance(Direct.DirectInterpreter, ParameterInterpreter)
        assert Direct.DirectInterpreter.name == "DirectInterpreter"

    def test_direct_interpreter_required_params(self):
        """Test DirectInterpreter required parameters."""
        assert "EncoderParams" in Direct.DirectInterpreter.requiredParams
        assert "FeedForwardParams" in Direct.DirectInterpreter.requiredParams
        assert "OutputDim" in Direct.DirectInterpreter.requiredParams
        assert Direct.DirectInterpreter.requiredParams["EncoderParams"] == dict
        assert Direct.DirectInterpreter.requiredParams["FeedForwardParams"] == dict
        assert Direct.DirectInterpreter.requiredParams["OutputDim"] == int

    def test_direct_scheduler_monitor(self, basic_params):
        """Test that scheduler monitors val_F1."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()

        assert optimizer_config['lr_scheduler']['monitor'] == 'val_F1'

    def test_direct_scheduler_factor(self, basic_params):
        """Test that scheduler reduces LR by correct factor."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.factor == 0.2

    def test_direct_scheduler_min_lr(self, basic_params):
        """Test that scheduler has minimum learning rate."""
        model = Direct(basic_params)
        optimizer_config = model.configure_optimizers()
        scheduler = optimizer_config['lr_scheduler']['scheduler']

        assert scheduler.min_lrs[0] == 5e-5

    def test_direct_invalid_encoder_params_type(self):
        """Test that Direct raises error for invalid EncoderParams type."""
        params = {
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
            "EncoderParams": "invalid_type",  # Should be dict
            "FeedForwardParams": {},
            "OutputDim": 10
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_invalid_feedforward_params_type(self):
        """Test that Direct raises error for invalid FeedForwardParams type."""
        params = {
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
            "EncoderParams": {},
            "FeedForwardParams": "invalid_type",  # Should be dict
            "OutputDim": 10
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_invalid_output_dim_type(self):
        """Test that Direct raises error for invalid OutputDim type."""
        params = {
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
            "EncoderParams": {},
            "FeedForwardParams": {},
            "OutputDim": "10"  # Should be int
        }

        with pytest.raises(TypeError):
            Direct(params)

    def test_direct_val_f1_metric_instance(self, basic_params):
        """Test that val_f1 metric is stored as instance variable."""
        model = Direct(basic_params)
        assert hasattr(model, 'val_f1')

    def test_direct_forward_adds_zero_column(self, basic_params):
        """Test that forward pass adds zero column."""
        model = Direct(basic_params)
        model.eval()
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 28, 28)
        global_features = torch.randn(batch_size, 1)
        
        predictions = model((time_series, global_features))
        
        # Should add a zero column
        assert predictions.shape == (batch_size, 10)
        # Last column should be zeros
        assert torch.allclose(predictions[:, -1], torch.zeros(batch_size))

    @pytest.mark.filterwarnings("ignore:You are trying to `self.log.*:UserWarning")
    def test_direct_training_step_unlabeled(self, basic_params):
        """Test training step with unlabeled data (None labels)."""
        model = Direct(basic_params)
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 28, 28)
        global_features = torch.randn(batch_size, 1)
        labels = None  # Unlabeled
        
        batch = ((time_series, global_features), labels)
        loss = model.training_step(batch, 0)
        
        # Loss should be 0 for unlabeled data
        assert loss == 0

    @pytest.mark.filterwarnings("ignore:You are trying to `self.log.*:UserWarning")
    def test_direct_validation_step_basic(self, basic_params):
        """Test validation step."""
        model = Direct(basic_params)
        
        batch_size = 4
        time_series = torch.randn(batch_size, 1, 28, 28)
        global_features = torch.randn(batch_size, 1)
        labels = torch.randint(0, 10, (batch_size,))
        
        batch = ((time_series, global_features), labels)
        result = model.validation_step(batch, 0)
        
        assert result is not None

    def test_direct_with_class_weights(self, basic_params):
        """Test Direct with class weights loaded from file."""
        import tempfile
        import yaml
        
        # Create temporary class weights file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            class_weights = {0: 1.0, 1: 1.5, 2: 2.0, 3: 1.2, 4: 1.3, 
                           5: 1.1, 6: 1.4, 7: 1.6, 8: 1.7, 9: 1.8}
            yaml.dump(class_weights, f)
            weights_path = f.name
        
        try:
            params = basic_params.copy()
            params["ClassWeightsPath"] = weights_path
            
            model = Direct(params)
            
            # Verify class weights were loaded
            assert hasattr(model, 'class_weights')
            assert model.class_weights is not None
            assert len(model.class_weights) == 10
            assert model.class_weights[0] == 1.0
            assert model.class_weights[2] == 2.0
        finally:
            import os
            os.unlink(weights_path)
    
    def test_direct_with_invalid_class_weights_path(self, basic_params, capsys):
        """Test Direct with invalid class weights path."""
        params = basic_params.copy()
        params["ClassWeightsPath"] = "/nonexistent/path/weights.yaml"
        
        model = Direct(params)
        
        # Should print error but not crash
        captured = capsys.readouterr()
        assert "Error" in captured.out or model.class_weights is None
    
    @pytest.mark.filterwarnings("ignore:You are trying to `self.log.*:UserWarning")
    def test_direct_training_step_with_class_weights(self, basic_params):
        """Test training step uses class weights when available."""
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            class_weights = {i: 1.0 + i * 0.1 for i in range(10)}
            yaml.dump(class_weights, f)
            weights_path = f.name
        
        try:
            params = basic_params.copy()
            params["ClassWeightsPath"] = weights_path
            model = Direct(params)
            
            batch_size = 4
            time_series = torch.randn(batch_size, 1, 28, 28)
            global_features = torch.randn(batch_size, 1)
            labels = torch.randint(0, 10, (batch_size,))
            
            batch = ((time_series, global_features), labels)
            loss = model.training_step(batch, 0)
            
            assert loss is not None
            assert loss > 0
        finally:
            import os
            os.unlink(weights_path)
    
    def test_direct_on_validation_epoch_end(self, basic_params):
        """Test validation epoch end resets F1 metric."""
        model = Direct(basic_params)
        
        # Call method - should not raise error
        model.on_validation_epoch_end()
        
        # Verify metric exists and is callable
        assert hasattr(model.val_f1, 'reset')
    
    def test_direct_configure_optimizers(self, basic_params):
        """Test optimizer configuration."""
        model = Direct(basic_params)
        
        result = model.configure_optimizers()
        
        assert result is not None
        # Returns a dict with optimizer and lr_scheduler
        assert isinstance(result, dict)
        assert 'optimizer' in result
        assert 'lr_scheduler' in result
        assert hasattr(result['optimizer'], 'step')
        assert hasattr(result['optimizer'], 'zero_grad')


