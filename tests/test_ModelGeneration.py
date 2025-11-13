"""
Unit tests for model generation in FinalPipeline.
Tests the model architecture setup without Optuna optimization.
Focuses on verifying encoder/decoder dimensions and forward pass compatibility.
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from GradientGang.Pipeline.Architectures.Direct import Direct
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Architectures.WindowedModelWrapper import WindowedModelWrapper


class MockTrial:
    """Mock Optuna trial for testing"""
    
    def __init__(self, params):
        self.params = params
        self.user_attrs = {}
        
    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])
    
    def suggest_int(self, name, low, high):
        return self.params.get(name, (low + high) // 2)
    
    def suggest_float(self, name, low, high, log=False):
        return self.params.get(name, (low + high) / 2)
    
    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class TestModelGeneration(unittest.TestCase):
    """Test model architecture generation"""
    
    def setUp(self):
        """Setup common test parameters"""
        self.batch_size = 2
        self.seq_len = 160
        self.num_channels = 34
        self.global_features = 6
        
        self.datasetInfo = {
            "timeSeriesShape": (self.num_channels, self.seq_len),
            "globalFeaturesShape": (self.global_features,)
        }
        
    def test_conv1d_autoencoder_dimensions(self):
        """Test Conv1d Autoencoder encoder-decoder dimension matching"""
        print("\n=== Testing Conv1d Autoencoder Dimensions ===")
        
        # Define architecture parameters
        archParams = {
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": 34,
                            "out_channels": 64,
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "bias": True,
                        }
                    },
                    {
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    },
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": 64,
                            "out_channels": 128,
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "bias": True,
                        }
                    },
                    {
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    },
                    {
                        "name": "AdaptiveAvgPool1d",
                        "params": {
                            "output_size": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {
                            "start_dim": 1,
                            "end_dim": -1,
                        }
                    }
                ]
            },
            "GlobalFFEncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 6,
                            "out_features": 32,
                            "bias": True,
                        }
                    }
                ]
            },
            "DecoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Unflatten",
                        "params": {
                            "dim": 1,
                            "unflattened_size": (128, 1),  # Should match encoder output
                        }
                    },
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": 128,
                            "out_channels": 64,
                            "kernel_size": 4,
                            "stride": 2,
                            "padding": 1,
                            "output_padding": 0,
                            "bias": True,
                        }
                    },
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": 64,
                            "out_channels": 34,
                            "kernel_size": 4,
                            "stride": 2,
                            "padding": 1,
                            "output_padding": 0,
                            "bias": True,
                        }
                    },
                    {
                        "name": "AdaptiveAvgPool1d",
                        "params": {
                            "output_size": 160,
                        }
                    }
                ]
            },
            "GlobalFFDecoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 32,
                            "out_features": 6,
                            "bias": True,
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
                            "in_features": 160,  # 128 (encoder) + 32 (global)
                            "out_features": 64,
                            "bias": True,
                        }
                    }
                ]
            },
            "OutputDim": 3,
            "LearningRate": 0.001,
            "RegularizationWeight": 0.01,
            "ReconstructionLossWeight": 0.5,
            "ClassWeightsPath": None,
        }
        
        # Create model
        model = LightningAutoencoder(archParams)
        
        # Create dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        globalFeatures = torch.randn(self.batch_size, self.global_features)
        labels = torch.randint(0, 3, (self.batch_size,))
        
        # Test forward pass
        print(f"Input shapes:")
        print(f"  timeSeries: {timeSeries.shape}")
        print(f"  globalFeatures: {globalFeatures.shape}")
        
        try:
            model.eval()
            with torch.no_grad():
                predictions, (decoded, decoded_global) = model((timeSeries, globalFeatures))
            
            print(f"Output shapes:")
            print(f"  predictions: {predictions.shape}")
            print(f"  decoded: {decoded.shape}")
            print(f"  decoded_global: {decoded_global.shape}")
            
            # Verify shapes
            self.assertEqual(predictions.shape, (self.batch_size, 3))
            self.assertEqual(decoded.shape, (self.batch_size, self.num_channels, self.seq_len))
            self.assertEqual(decoded_global.shape, (self.batch_size, self.global_features))
            
            print("✓ Conv1d Autoencoder forward pass successful!")
            
        except Exception as e:
            print(f"✗ Conv1d Autoencoder forward pass failed!")
            print(f"Error: {e}")
            raise
    
    def test_lstm_autoencoder_dimensions(self):
        """Test LSTM Autoencoder encoder-decoder dimension matching"""
        print("\n=== Testing LSTM Autoencoder Dimensions ===")
        
        # Define architecture parameters
        archParams = {
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "LSTM",
                        "params": {
                            "input_size": 34,
                            "hidden_size": 64,
                            "num_layers": 2,
                            "bias": True,
                            "batch_first": True,
                            "dropout": 0.2,
                            "bidirectional": True,
                        }
                    }
                ]
            },
            "GlobalFFEncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 6,
                            "out_features": 32,
                            "bias": True,
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
                            "num_layers": 2,
                            "bias": True,
                            "batch_first": True,
                            "dropout": 0.2,
                            "bidirectional": False,
                        }
                    },
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 64,
                            "out_features": 34,
                            "bias": True,
                        }
                    }
                ]
            },
            "GlobalFFDecoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 32,
                            "out_features": 6,
                            "bias": True,
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
                            "in_features": 160,  # 128 (encoder) + 32 (global)
                            "out_features": 64,
                            "bias": True,
                        }
                    }
                ]
            },
            "OutputDim": 3,
            "LearningRate": 0.001,
            "RegularizationWeight": 0.01,
            "ReconstructionLossWeight": 0.5,
            "ClassWeightsPath": None,
        }
        
        # Create model
        model = LightningAutoencoder(archParams)
        
        # Create dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        globalFeatures = torch.randn(self.batch_size, self.global_features)
        
        # Test forward pass
        print(f"Input shapes:")
        print(f"  timeSeries: {timeSeries.shape}")
        print(f"  globalFeatures: {globalFeatures.shape}")
        
        try:
            model.eval()
            with torch.no_grad():
                predictions, (decoded, decoded_global) = model((timeSeries, globalFeatures))
            
            print(f"Output shapes:")
            print(f"  predictions: {predictions.shape}")
            print(f"  decoded: {decoded.shape}")
            print(f"  decoded_global: {decoded_global.shape}")
            
            # Verify shapes
            self.assertEqual(predictions.shape, (self.batch_size, 3))
            self.assertEqual(decoded.shape, (self.batch_size, self.num_channels, self.seq_len))
            self.assertEqual(decoded_global.shape, (self.batch_size, self.global_features))
            
            print("✓ LSTM Autoencoder forward pass successful!")
            
        except Exception as e:
            print(f"✗ LSTM Autoencoder forward pass failed!")
            print(f"Error: {e}")
            raise
    
    def test_conv1d_direct_model(self):
        """Test Conv1d Direct model"""
        print("\n=== Testing Conv1d Direct Model ===")
        
        archParams = {
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": 34,
                            "out_channels": 64,
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "bias": True,
                        }
                    },
                    {
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    },
                    {
                        "name": "AdaptiveAvgPool1d",
                        "params": {
                            "output_size": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {
                            "start_dim": 1,
                            "end_dim": -1,
                        }
                    }
                ]
            },
            "GlobalFFEncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 6,
                            "out_features": 32,
                            "bias": True,
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
                            "in_features": 96,  # 64 (encoder) + 32 (global)
                            "out_features": 64,
                            "bias": True,
                        }
                    }
                ]
            },
            "OutputDim": 3,
            "LearningRate": 0.001,
            "RegularizationWeight": 0.01,
            "ClassWeightsPath": None,
        }
        
        # Create model
        model = Direct(archParams)
        
        # Create dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        globalFeatures = torch.randn(self.batch_size, self.global_features)
        
        # Test forward pass
        print(f"Input shapes:")
        print(f"  timeSeries: {timeSeries.shape}")
        print(f"  globalFeatures: {globalFeatures.shape}")
        
        try:
            model.eval()
            with torch.no_grad():
                predictions = model((timeSeries, globalFeatures))
            
            print(f"Output shape: {predictions.shape}")
            
            # Verify shapes
            self.assertEqual(predictions.shape, (self.batch_size, 3))
            
            print("✓ Conv1d Direct model forward pass successful!")
            
        except Exception as e:
            print(f"✗ Conv1d Direct model forward pass failed!")
            print(f"Error: {e}")
            raise
    
    def test_windowed_wrapper_with_conv1d_autoencoder(self):
        """Test WindowedModelWrapper with Conv1d Autoencoder"""
        print("\n=== Testing WindowedModelWrapper with Conv1d Autoencoder ===")
        
        # Define architecture parameters (same as test_conv1d_autoencoder_dimensions)
        archParams = {
            "EncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": 34,
                            "out_channels": 64,
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "bias": True,
                        }
                    },
                    {
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    },
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": 64,
                            "out_channels": 128,
                            "kernel_size": 5,
                            "stride": 1,
                            "padding": 2,
                            "bias": True,
                        }
                    },
                    {
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    },
                    {
                        "name": "AdaptiveAvgPool1d",
                        "params": {
                            "output_size": 1,
                        }
                    },
                    {
                        "name": "Flatten",
                        "params": {
                            "start_dim": 1,
                            "end_dim": -1,
                        }
                    }
                ]
            },
            "GlobalFFEncoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 6,
                            "out_features": 32,
                            "bias": True,
                        }
                    }
                ]
            },
            "DecoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Unflatten",
                        "params": {
                            "dim": 1,
                            "unflattened_size": (128, 1),
                        }
                    },
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": 128,
                            "out_channels": 64,
                            "kernel_size": 4,
                            "stride": 2,
                            "padding": 1,
                            "output_padding": 0,
                            "bias": True,
                        }
                    },
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": 64,
                            "out_channels": 34,
                            "kernel_size": 4,
                            "stride": 2,
                            "padding": 1,
                            "output_padding": 0,
                            "bias": True,
                        }
                    },
                    {
                        "name": "AdaptiveAvgPool1d",
                        "params": {
                            "output_size": 160,
                        }
                    }
                ]
            },
            "GlobalFFDecoderParams": {
                "activation_function": "ReLU",
                "layer_type": [
                    {
                        "name": "Linear",
                        "params": {
                            "in_features": 32,
                            "out_features": 6,
                            "bias": True,
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
                            "in_features": 160,
                            "out_features": 64,
                            "bias": True,
                        }
                    }
                ]
            },
            "OutputDim": 3,
            "LearningRate": 0.001,
            "RegularizationWeight": 0.01,
            "ReconstructionLossWeight": 0.5,
            "ClassWeightsPath": None,
        }
        
        # Create base model
        base_model = LightningAutoencoder(archParams)
        
        # Wrap with WindowedModelWrapper
        window_size = 80
        stride = 40
        model = WindowedModelWrapper(
            base_model=base_model,
            window_size=window_size,
            stride=stride,
            aggregation_method="avg_logits",
            window_loss_weight=0.0
        )
        
        # Create dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        globalFeatures = torch.randn(self.batch_size, self.global_features)
        labels = torch.randint(0, 3, (self.batch_size,))
        
        # Test forward pass
        print(f"Input shapes:")
        print(f"  timeSeries: {timeSeries.shape}")
        print(f"  globalFeatures: {globalFeatures.shape}")
        print(f"Window size: {window_size}, stride: {stride}")
        
        try:
            model.eval()
            with torch.no_grad():
                output = model((timeSeries, globalFeatures))
            
            if isinstance(output, tuple):
                predictions, reconstructions = output
                print(f"Output shapes:")
                print(f"  predictions: {predictions.shape}")
                print(f"  reconstructions: ({reconstructions[0].shape}, {reconstructions[1].shape})")
            else:
                predictions = output
                print(f"Output shape: {predictions.shape}")
            
            # Verify shapes - predictions should be sample-level
            self.assertEqual(predictions.shape, (self.batch_size, 3))
            
            print("✓ WindowedModelWrapper with Conv1d Autoencoder forward pass successful!")
            
        except Exception as e:
            print(f"✗ WindowedModelWrapper forward pass failed!")
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
