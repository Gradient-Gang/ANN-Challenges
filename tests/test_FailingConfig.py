"""
Test with the EXACT failing configuration from the user.
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from GradientGang.Pipeline.FinalPipeline import FinalPipeline


class MockTrialFailingConfig:
    """Mock trial with the exact failing configuration"""
    
    def __init__(self):
        self.params = {
            'MacroArchitecture': 'Autoencoder',
            'use_windowing': True,
            'window_size': 9,
            'stride_ratio': 0.75,
            'aggregation_method': 'majority_vote',
            'window_loss_weight': 0.106169555339138,
            
            # Global encoder
            'globalEmbeddingDim': 36,
            'globalNumLayers': 1,
            'globalDropout': 0.152121121479769,
            'globalActivation': 'ReLU',
            
            # Time series encoder - RECURRENT (GRU)
            'architectureType': 'Recurrent',
            'rnnType': 'GRU',
            'hiddenDim': 205,
            'numLayers': 1,
            'bidirectional': True,
            'recurrentDropout': 0.0232252063599989,
            'encoderActivation': 'ReLU',
            
            # Feedforward
            'numFFLayers': 3,
            'ffHiddenDim': 249,
            'ffDropout': 0.404198674058231,
            'ffActivation': 'GELU',
            
            # Training
            'ReconstructionLossWeight': 0.452121994991681,
            'LearningRate': 2.32335035153901e-05,
            'RegularizationWeight': 0.0956549921594382,
            'SchedulerType': 'CosineAnnealing',
            'eta_min': 2.11370594406457e-06,
            'EarlyStoppingPatience': 13,
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


class TestFailingConfiguration(unittest.TestCase):
    """Test with the exact failing configuration"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 160
        self.num_channels = 34
        self.global_features = 6
        
        self.datasetInfo = {
            "timeSeriesShape": (self.num_channels, self.seq_len),
            "globalFeaturesShape": (self.global_features,)
        }
    
    def test_failing_config_architecture_setup(self):
        """Test architecture setup with the failing configuration"""
        print("\n=== Testing Failing Configuration ===")
        print("Architecture: Recurrent (GRU)")
        print("hiddenDim: 205, bidirectional: True")
        print("Expected encoder output: 205 * 2 = 410")
        
        # Create mock trial
        trial = MockTrialFailingConfig()
        
        # Create FinalPipeline and setup architecture
        pipeline = object.__new__(FinalPipeline)
        
        archParams = {}
        archParams = pipeline.setUpEncoder(trial, archParams, self.datasetInfo)
        archParams = pipeline.setUpFeedForwardHead(trial, archParams, self.datasetInfo)
        
        # Check encoder
        encoder_params = archParams["EncoderParams"]
        print(f"\nEncoder parameters:")
        print(f"  Architecture type: {trial.params['architectureType']}")
        print(f"  First layer: {encoder_params['layer_type'][0]['name']}")
        print(f"  Number of layers in list: {len(encoder_params['layer_type'])}")
        
        # Now setup decoder
        print("\nCalling setUpDecoder...")
        archParams = pipeline.setUpDecoder(trial, archParams, self.datasetInfo)
        
        decoder_params = archParams["DecoderParams"]
        print(f"\nDecoder parameters:")
        print(f"  Number of layers: {len(decoder_params['layer_type'])}")
        print(f"  First layer: {decoder_params['layer_type'][0]['name']}")
        
        # Check if first layer is Unflatten (BUG!) or RNN (correct)
        first_decoder_layer = decoder_params['layer_type'][0]['name']
        
        if first_decoder_layer == 'Unflatten':
            print(f"\n❌ BUG FOUND!")
            print(f"   Decoder first layer is 'Unflatten', but encoder is RNN!")
            print(f"   This will cause shape mismatch errors!")
            print(f"   Unflatten params: {decoder_params['layer_type'][0]['params']}")
            self.fail("Decoder is using Unflatten for RNN encoder!")
        elif first_decoder_layer in ['LSTM', 'GRU', 'RNN']:
            print(f"\n✓ Decoder correctly uses RNN layer: {first_decoder_layer}")
        else:
            print(f"\n⚠️ Unexpected decoder first layer: {first_decoder_layer}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
