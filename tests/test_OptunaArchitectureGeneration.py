"""
Test to reproduce the exact bug from Optuna trial generation.
This test simulates how FinalPipeline generates architecture parameters.
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from GradientGang.Pipeline.FinalPipeline import FinalPipeline
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder


class MockTrial:
    """Mock Optuna trial for testing"""
    
    def __init__(self):
        # Use fixed hyperparameters that would trigger the bug
        self.params = {
            'MacroArchitecture': 'Autoencoder',
            'use_windowing': True,
            'window_size': 80,
            'stride_ratio': 0.5,
            'aggregation_method': 'avg_logits',
            'window_loss_weight': 0.2,
            
            # Global encoder
            'globalEmbeddingDim': 64,
            'globalNumLayers': 2,
            'globalDropout': 0.2,
            'globalActivation': 'ReLU',
            'globalHiddenDim_0': 48,
            
            # Time series encoder - MultiScaleCNN (might cause 832 channels)
            'architectureType': 'MultiScaleCNN',
            'numMultiScaleLayers': 2,
            'useDilation': False,
            'kernelSizesIndex': 2,  # (5, 9, 13)
            'encoderActivation': 'ReLU',
            'multiscaleBranchChannels_0': 69,  # 69 * 3 branches = 207
            'multiscalePoolType_0': 'max',
            'multiscaleBranchChannels_1': 139,  # 139 * 3 branches = 417, but after conv might be different
            'multiscalePoolType_1': 'max',
            
            # Feedforward
            'numFFLayers': 2,
            'ffHiddenDim': 128,
            'ffDropout': 0.3,
            'ffActivation': 'ReLU',
            
            # Training
            'ReconstructionLossWeight': 0.5,
            'LearningRate': 0.001,
            'RegularizationWeight': 0.01,
            'SchedulerType': 'ReduceLROnPlateau',
            'SchedulerPatience': 10,
            'SchedulerFactor': 0.5,
            'SchedulerMinLR': 1e-5,
            'EarlyStoppingPatience': 15,
        }
        self.user_attrs = {}
        
    def suggest_categorical(self, name, choices):
        if name in self.params:
            return self.params[name]
        return choices[0]
    
    def suggest_int(self, name, low, high):
        if name in self.params:
            return self.params[name]
        return (low + high) // 2
    
    def suggest_float(self, name, low, high, log=False):
        if name in self.params:
            return self.params[name]
        return (low + high) / 2
    
    def set_user_attr(self, key, value):
        self.user_attrs[key] = value
    
    def report(self, value, step):
        pass
    
    def should_prune(self):
        return False


class TestOptunaArchitectureGeneration(unittest.TestCase):
    """Test architecture generation as done by Optuna"""
    
    def setUp(self):
        """Setup test pipeline"""
        self.batch_size = 2
        self.seq_len = 160
        self.num_channels = 34
        self.global_features = 6
        
        self.datasetInfo = {
            "timeSeriesShape": (self.num_channels, self.seq_len),
            "globalFeaturesShape": (self.global_features,)
        }
        
        # Create a minimal FinalPipeline instance
        self.pipeline_params = {
            "database_path": "../src/GradientGang/Pipeline/Optimizer/.env",
            "data_params": {
                'data_dir': "../dataset/PirateProcessed/",
                'train_file_name': "pirate_pain_train.csv",
                'train_file_name_labels': "pirate_pain_train_labels.csv",
                'test_file_name': "pirate_pain_test.csv",
                'train_global_features_file': "train_global_features.csv",
                'test_global_features_file': "test_global_features.csv",
                'batch_size': 32,
                'num_workers': 0,
                'val_split': 0.2,
                'shuffle': True,
                'use_kfold': True,
                'n_folds': 4,
            },
            'project_name': 'test_architecture_generation',
            'study_name': 'test_bug_reproduction'
        }
    
    def test_finalpipeline_setup_encoder(self):
        """Test FinalPipeline.setUpEncoder method directly"""
        print("\n=== Testing FinalPipeline.setUpEncoder Method ===")
        
        # Create mock trial
        trial = MockTrial()
        
        # Create a FinalPipeline instance (we'll skip initialization that requires DB)
        from GradientGang.Pipeline.FinalPipeline import FinalPipeline
        
        # Create a minimal pipeline object
        pipeline = object.__new__(FinalPipeline)
        
        # Call setUpEncoder
        archParams = {}
        archParams = pipeline.setUpEncoder(trial, archParams, self.datasetInfo)
        
        # Print the encoder parameters
        print("\nEncoder Parameters:")
        encoder_params = archParams["EncoderParams"]
        print(f"  Activation: {encoder_params['activation_function']}")
        print(f"  Number of layers: {len(encoder_params['layer_type'])}")
        for i, layer in enumerate(encoder_params['layer_type']):
            print(f"    {i}: {layer['name']}")
        
        # Now test the encoder
        from GradientGang.Pipeline.Architectures.Encoder import Encoder
        
        encoder = Encoder(
            params=encoder_params,
            num_input_channels=34,
            base_channel_size=64,
            latent_dim=128
        )
        
        # Test with dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        
        print(f"\nInput shape: {timeSeries.shape}")
        
        with torch.no_grad():
            encoded = encoder(timeSeries)
        
        print(f"Encoded shape: {encoded.shape}")
        
        # Check what the expected output size should be
        first_layer = encoder_params['layer_type'][0]
        if first_layer['name'] == 'Conv1d':
            # Find last Conv1d layer
            last_conv_channels = None
            for layer in encoder_params['layer_type']:
                if layer['name'] == 'Conv1d':
                    last_conv_channels = layer['params']['out_channels']
            
            print(f"Expected shape: ({self.batch_size}, {last_conv_channels})")
            
            # Check if AdaptiveAvgPool1d is present
            has_adaptive_pool = any(layer['name'] == 'AdaptiveAvgPool1d' for layer in encoder_params['layer_type'])
            has_flatten = any(layer['name'] == 'Flatten' for layer in encoder_params['layer_type'])
            
            print(f"Has AdaptiveAvgPool1d: {has_adaptive_pool}")
            print(f"Has Flatten: {has_flatten}")
            
            if not has_adaptive_pool:
                print("❌ BUG FOUND! Missing AdaptiveAvgPool1d layer!")
            if not has_flatten:
                print("❌ BUG FOUND! Missing Flatten layer!")
            
            if has_adaptive_pool and has_flatten:
                self.assertEqual(encoded.shape, (self.batch_size, last_conv_channels),
                               f"Encoder should output (batch, {last_conv_channels}) after AdaptiveAvgPool1d(1) and Flatten")
        
        elif first_layer['name'] in ['LSTM', 'GRU', 'RNN']:
            hidden_size = first_layer['params']['hidden_size']
            bidirectional = first_layer['params']['bidirectional']
            expected_size = hidden_size * (2 if bidirectional else 1)
            print(f"Expected shape: ({self.batch_size}, {expected_size})")
            self.assertEqual(encoded.shape, (self.batch_size, expected_size))
    
    def test_finalpipeline_setup_decoder(self):
        """Test FinalPipeline.setUpDecoder method directly"""
        print("\n=== Testing FinalPipeline.setUpDecoder Method ===")
        
        # Create mock trial
        trial = MockTrial()
        
        # Create a FinalPipeline instance
        from GradientGang.Pipeline.FinalPipeline import FinalPipeline
        
        pipeline = object.__new__(FinalPipeline)
        
        # First setup encoder
        archParams = {}
        archParams = pipeline.setUpEncoder(trial, archParams, self.datasetInfo)
        archParams = pipeline.setUpFeedForwardHead(trial, archParams, self.datasetInfo)
        
        # Now setup decoder
        archParams = pipeline.setUpDecoder(trial, archParams, self.datasetInfo)
        
        # Print the decoder parameters
        print("\nDecoder Parameters:")
        decoder_params = archParams["DecoderParams"]
        print(f"  Activation: {decoder_params['activation_function']}")
        print(f"  Number of layers: {len(decoder_params['layer_type'])}")
        for i, layer in enumerate(decoder_params['layer_type']):
            print(f"    {i}: {layer['name']}", end="")
            if layer['name'] == 'Unflatten':
                print(f" -> unflattened_size={layer['params']['unflattened_size']}", end="")
            elif 'out_channels' in layer.get('params', {}):
                print(f" -> out_channels={layer['params']['out_channels']}", end="")
            print()
        
        # Get encoder output channels
        encoder_params = archParams["EncoderParams"]
        first_layer = encoder_params['layer_type'][0]
        
        if first_layer['name'] == 'Conv1d':
            conv_layers = [l for l in encoder_params['layer_type'] if l['name'] == 'Conv1d']
            last_conv_channels = conv_layers[-1]['params']['out_channels']
            print(f"\nEncoder last Conv1d out_channels: {last_conv_channels}")
        elif first_layer['name'] == 'MultiScaleCNN':
            multiscale_layers = [l for l in encoder_params['layer_type'] if l['name'] == 'MultiScaleCNN']
            last_layer = multiscale_layers[-1]
            branch_channels = last_layer['params']['branch_channels']
            num_branches = len(last_layer['params']['kernel_sizes'])
            last_conv_channels = branch_channels * num_branches
            print(f"\nEncoder last MultiScaleCNN out_channels: {last_conv_channels} ({branch_channels} x {num_branches} branches)")
        elif first_layer['name'] in ['LSTM', 'GRU', 'RNN']:
            hidden_size = first_layer['params']['hidden_size']
            bidirectional = first_layer['params']['bidirectional']
            last_conv_channels = hidden_size * (2 if bidirectional else 1)
            print(f"\nEncoder {first_layer['name']} output size: {last_conv_channels}")
        else:
            print(f"\n⚠️ Unknown encoder type: {first_layer['name']}")
            return
        
        # Check the Unflatten layer
        unflatten_layer = decoder_params['layer_type'][0]
        if unflatten_layer['name'] == 'Unflatten':
            unflattened_size = unflatten_layer['params']['unflattened_size']
            print(f"Decoder Unflatten size: {unflattened_size}")
            
            expected_unflatten = (last_conv_channels, 1)
            print(f"Expected Unflatten size: {expected_unflatten}")
            
            if unflattened_size != expected_unflatten:
                print(f"❌ BUG FOUND! Unflatten size mismatch!")
                print(f"   Got: {unflattened_size}")
                print(f"   Expected: {expected_unflatten}")
            else:
                print(f"✓ Unflatten size is correct!")
    
    def test_conv1d_autoencoder_architecture_generation(self):
        """Test Conv1d Autoencoder architecture generation reproduces the bug"""
        print("\n=== Testing Conv1d Autoencoder Architecture Generation ===")
        
        # Create mock trial
        trial = MockTrial()
        
        # Create a temporary FinalPipeline to use its setup methods
        # We'll just use the setup methods without the full pipeline
        archParams = {}
        
        # Simulate the setup process
        from GradientGang.Pipeline.FinalPipeline import FinalPipeline
        
        # We need to create a minimal pipeline object to access the methods
        # But we can't initialize it fully, so let's directly call the setup logic
        
        # Instead, let's manually build what setUpEncoder would create
        print("Building encoder parameters...")
        
        # Global encoder params
        globalEmbeddingDim = 64
        globalLayerList = [
            {
                "name": "Linear",
                "params": {
                    "in_features": 6,
                    "out_features": 48,
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
                    "in_features": 48,
                    "out_features": 64,
                    "bias": True,
                }
            }
        ]
        
        globalEncoderParams = {
            "activation_function": "ReLU",
            "layer_type": globalLayerList,
        }
        
        # Time series encoder params - Conv1d with 2 layers
        layerList = []
        inputChannels = 34
        currentChannels = inputChannels
        
        # Layer 0
        outChannels = 64
        layerList.append({
            "name": "Conv1d",
            "params": {
                "in_channels": currentChannels,
                "out_channels": outChannels,
                "kernel_size": 5,
                "stride": 1,
                "padding": 2,
                "bias": True,
            }
        })
        layerList.append({
            "name": "MaxPool1d",
            "params": {
                "kernel_size": 2,
                "stride": 2,
            }
        })
        currentChannels = outChannels
        
        # Layer 1
        outChannels = 128
        layerList.append({
            "name": "Conv1d",
            "params": {
                "in_channels": currentChannels,
                "out_channels": outChannels,
                "kernel_size": 5,
                "stride": 1,
                "padding": 2,
                "bias": True,
            }
        })
        layerList.append({
            "name": "MaxPool1d",
            "params": {
                "kernel_size": 2,
                "stride": 2,
            }
        })
        currentChannels = outChannels
        
        # Add adaptive pooling and flatten
        layerList.append({
            "name": "AdaptiveAvgPool1d",
            "params": {
                "output_size": 1,
            }
        })
        layerList.append({
            "name": "Flatten",
            "params": {
                "start_dim": 1,
                "end_dim": -1,
            }
        })
        
        timeSeriesEncoderParams = {
            "activation_function": "ReLU",
            "layer_type": layerList,
        }
        
        print(f"Encoder layers: {len(layerList)}")
        for i, layer in enumerate(layerList):
            print(f"  {i}: {layer['name']}")
        
        archParams["EncoderParams"] = timeSeriesEncoderParams
        archParams["GlobalFFEncoderParams"] = globalEncoderParams
        
        # Now test the encoder to see what output shape we get
        from GradientGang.Pipeline.Architectures.Encoder import Encoder
        
        encoder = Encoder(
            params=timeSeriesEncoderParams,
            num_input_channels=34,
            base_channel_size=64,
            latent_dim=128
        )
        
        # Test with dummy input
        timeSeries = torch.randn(self.batch_size, self.num_channels, self.seq_len)
        
        print(f"\nInput shape: {timeSeries.shape}")
        
        with torch.no_grad():
            encoded = encoder(timeSeries)
        
        print(f"Encoded shape: {encoded.shape}")
        print(f"Expected shape: ({self.batch_size}, 128)")
        
        # Check if shape is correct
        if encoded.shape != (self.batch_size, 128):
            print(f"❌ BUG REPRODUCED! Encoder output shape is wrong!")
            print(f"   Got: {encoded.shape}")
            print(f"   Expected: ({self.batch_size}, 128)")
            
            # Calculate what went wrong
            if len(encoded.shape) == 3:
                print(f"   Encoder output is still 3D: (batch={encoded.shape[0]}, channels={encoded.shape[1]}, seq_len={encoded.shape[2]})")
                print(f"   This means AdaptiveAvgPool1d or Flatten didn't work correctly!")
            elif len(encoded.shape) == 2:
                expected_features = encoded.shape[1]
                print(f"   Encoder output features: {expected_features}")
                if expected_features > 128:
                    seq_len_before_flatten = expected_features / 128
                    print(f"   Implied sequence length before flatten: {seq_len_before_flatten}")
                    print(f"   AdaptiveAvgPool1d(1) was NOT applied correctly!")
        else:
            print(f"✓ Encoder output shape is correct!")
        
        self.assertEqual(encoded.shape, (self.batch_size, 128), 
                        "Encoder should output (batch, 128) after AdaptiveAvgPool1d(1) and Flatten")


if __name__ == "__main__":
    unittest.main(verbosity=2)
