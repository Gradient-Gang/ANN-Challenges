from GradientGang.Pipeline.SubmissionGenerator.WindowedSubmissionGenerator import (
    WindowedSubmissionGenerator,
)
from GradientGang.Pipeline.Utils.EnsembleModels import EnsembleModel
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter
from GradientGang.Pipeline.Architectures.WindowedModelWrapper import (
    WindowedModelWrapper,
)
from GradientGang.Pipeline.Architectures.LightningAutoencoder import (
    LightningAutoencoder,
)
from GradientGang.Pipeline.Architectures.Direct import Direct
from GradientGang.Pipeline.DataLoader.DataLoader import DataModule
import optuna
import torch
import copy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import F1Score
import warnings
import dotenv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from datetime import datetime

warnings.filterwarnings("ignore")


class FinalPipeline:

    def __init__(self, params: dict):
        self.params = params
        self.storage = None
        self.dataloader = None
        self.study = None
        self.best_model = None

        # Initialize project namee
        self.project_name = params.get("project_name", None)
        if not self.project_name:
            raise ValueError("Project name must be provided in parameters.")

        # Initialize data parameters
        self.data_params = params.get("data_params", None)
        if not self.data_params:
            raise ValueError("Data parameters must be provided in parameters.")

        # Initialize submission folder
        self.submission_folder = params.get("submission_path", None)
        if not self.submission_folder:
            raise ValueError("Submission path must be provided in parameters.")

        # Initialize database connection
        self.database_url = params.get("database_url", None)
        if not self.database_url:
            raise ValueError("Database URL must be provided in parameters.")
        self.load_database()
        print("✓ Database initialized successfully!")

        # Initialize study
        self.study_name = params.get("study_name", None)
        if not self.study_name:
            raise ValueError("Study name must be provided in parameters.")
        self.create_study()
        print("✓ Study initialized successfully!")
        print("✓ FinalPipeline initialized successfully!")

    # DB management
    def load_database(self):
        """
        Load database configuration from environment file.

        This URL is used for persistent storage of Optuna study results.
        """
        self.storage = self.database_url
        print(
            f"Storage: {self.storage[:21]}..."
            if self.storage
            else "⚠️ No database URL found"
        )
        print("✓ Database configuration loaded")

    def create_study(self):
        """
        Create or load an Optuna study for hyperparameter optimization.

        Configures an Optuna study with:
        - Direction: Maximize (optimizing for F1 score)
        - Sampler: TPESampler with fixed seed for reproducibility
        - Pruner: MedianPruner for early stopping of unpromising trials
        - Storage: PostgreSQL database for persistent results
        - Resume capability: Loads existing study if available

        The study name is constructed from project_name + study_name to ensure
        uniqueness across different experiments.
        """
        self.study = optuna.create_study(
            direction="maximize",  # Maximize F1 score
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner = optuna.pruners.PatientPruner(
                optuna.pruners.MedianPruner(
                    n_startup_trials=10,
                    n_warmup_steps=1
                ),
                patience=1  # Bad for 2 consecutive folds → prune
            ),
            study_name=self.project_name + self.study_name,
            storage=self.storage,
            load_if_exists=True,  # Resume from existing study if available
        )

        print("✓ Study created/loaded successfully!")
        print(f"Study name: {self.study.study_name}")
        print(f"Sampler: {self.study.sampler.__class__.__name__}")
        print(f"Pruner: {self.study.pruner.__class__.__name__}")
        print(f"Storage: {'Database' if self.storage else 'In-memory'}")
        print(f"Total trials: {len(self.study.trials)}")

    def study_summary(self):
        """
        Print the current status of the Optuna study.
        Displays key information about the study including:
        - Study name
        - Direction (maximize/minimize)
        - Total number of trials
        - Number of completed, failed, pruned, and running trials
        - Best trial number and F1 score
        - Top 5 trials with architecture details
        ️- Recent trial states with symbols indicating status
        ️- Warning if no trials exist in the study.
        """

        study = self.study
        print(f"Study name: {study.study_name}")
        print(f"Direction: {study.direction}")
        print(f"Total trials: {len(study.trials)}")
        print(
            f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        print(
            f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}"
        )
        print(
            f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        print(
            f"Running trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}"
        )

        if len(study.trials) > 0:
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if completed_trials:
                print(f"\n✓ Best trial: {study.best_trial.number}")
                print(f"✓ Best F1 score: {study.best_value:.4f}")
                print(f"\nTop 5 trials:")
                sorted_trials = sorted(
                    completed_trials, key=lambda t: t.value, reverse=True
                )[:5]
                for i, trial in enumerate(sorted_trials, 1):
                    arch = trial.params.get("MacroArchitecture", "Autoencoder")
                    enc = trial.params.get("architectureType", "Unknown")
                    print(
                        f"  {i}. Trial {trial.number}: F1={trial.value:.4f} | {arch} | {enc}"
                    )

            print("\n📊 Trial states (last 10):")
            for trial in study.trials[-10:]:
                state_symbol = (
                    "✓"
                    if trial.state == optuna.trial.TrialState.COMPLETE
                    else (
                        "✗"
                        if trial.state == optuna.trial.TrialState.FAIL
                        else (
                            "⊗"
                            if trial.state == optuna.trial.TrialState.PRUNED
                            else "⟳"
                        )
                    )
                )
                value_str = (
                    f"F1={trial.value:.4f}" if trial.value is not None else "N/A"
                )
                print(
                    f"  {state_symbol} Trial {trial.number}: {trial.state.name} | {value_str}"
                )
        else:
            print("\n⚠️ No trials found in this study. Run optimization to start!")

    # optimizer
    def optuna_optimize(self, n_trials: int = 500):
        """
        Run Optuna hyperparameter optimization with K-fold cross-validation.

        This method starts the optimization process, running the specified number
        of trials. Each trial:
        1. Suggests hyperparameters (architecture, learning rate, etc.)
        2. Trains models using K-fold cross-validation
        3. Reports mean F1 score across folds
        4. May be pruned if performance is poor

        Args:
            n_trials (int, optional): Number of optimization trials to run.

        Example:
            >>> pipeline.optuna_optimize(n_trials=100)
        """
        print(f"Starting optimization ...")
        print("This may take a while depending on your hardware.")
        print("-" * 60)

        self.study.optimize(
            self.objective_kfold, show_progress_bar=True, n_trials=n_trials
        )

        print("\nOptimization completed!")
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best F1 score: {self.study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

    # Architecture Setup Methods
    def setUpEncoder(
        self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict
    ):
        """
        Configure encoder architecture based on Optuna trial suggestions.

        This method sets up both:
        1. Global feature encoder (feedforward layers for tabular features)
        2. Time series encoder (Recurrent/Conv1d/MultiScaleCNN for sequential data)

        Supported time series encoder types:
        - Recurrent: LSTM/GRU with configurable layers, hidden dimensions, bidirectionality
        - Conv1d: 1D convolutional layers with pooling for sequence processing
        - MultiScaleCNN: Inception-style multi-scale convolutions for multi-resolution features

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter suggestions
            architectureParameters (dict): Dictionary to store architecture configuration
            datasetInfo (dict): Dataset metadata containing:
                - globalFeaturesShape: Shape of global/tabular features
                - timeSeriesShape: Shape of time series data (channels, length)

        Returns:
            dict: Updated architectureParameters with added keys:
                - GlobalFFEncoderParams: Configuration for global feature encoder
                - EncoderParams: Configuration for time series encoder

        Note:
            The encoder output is flattened and will be concatenated with global
            feature embeddings before being passed to the feedforward head.
        """
        # First setup global feature encoder
        globalInputDim = datasetInfo["globalFeaturesShape"][0]
        globalEmbeddingDim = trial.suggest_int("globalEmbeddingDim", 16, 40)
        globalNumLayers = trial.suggest_int("globalNumLayers", 1, 4)
        globalDropout = trial.suggest_float("globalDropout", 0.0, 0.4)
        globalActivation = "LeakyReLU"  # FIXED VALUE
        # trial.suggest_categorical(
        #     "globalActivation", ["ReLU", "LeakyReLU", "GELU"]
        # )

        # Build global feature encoder layers
        globalLayerList = []
        currentDim = globalInputDim
        for i in range(globalNumLayers):
            nextDim = (
                globalEmbeddingDim
                if i == globalNumLayers - 1
                else trial.suggest_int(f"globalHiddenDim_{i}", 10, 128)
            )
            globalLayerList.append(
                {
                    "name": "Linear",
                    "params": {
                        "in_features": currentDim,
                        "out_features": nextDim,
                        "bias": True,
                    },
                }
            )
            if globalDropout > 0 and i < globalNumLayers - 1:
                globalLayerList.append(
                    {
                        "name": "Dropout",
                        "params": {
                            "p": globalDropout,
                            "inplace": False,
                        },
                    }
                )
            currentDim = nextDim

        globalEncoderParams = {
            "activation_function": globalActivation,
            "layer_type": globalLayerList,
        }
        architectureParameters["GlobalFFEncoderParams"] = globalEncoderParams

        # Setup time series encoder - now supports Recurrent, Conv1d, and MultiScaleCNN
        architectureType = trial.suggest_categorical(
            "architectureType", ["Recurrent", "MultiScaleCNN"]
        )

        timeSeriesEncoderParams = {}

        if architectureType == "Recurrent":
            rnnType = trial.suggest_categorical("rnnType", ["LSTM", "GRU"])
            hiddenDim = trial.suggest_int("hiddenDim", 100, 300)
            numLayers = trial.suggest_int("RecurrentNumLayers", 1, 4)
            bidirectional = True # FIXED VALUE trial.suggest_categorical("bidirectional", [False, True])
            dropout = trial.suggest_float("recurrentDropout", 0.0, 0.5)
            activationFunction = "GELU" # FIXED VALUE
            # trial.suggest_categorical(
            #     "encoderActivation", ["ReLU", "LeakyReLU", "GELU"]
            # )

            # Get input size from dataset info
            inputSize = datasetInfo["timeSeriesShape"][0]

            timeSeriesEncoderParams = {
                "activation_function": activationFunction,
                "layer_type": [
                    {
                        "name": rnnType,
                        "params": {
                            "input_size": inputSize,
                            "hidden_size": hiddenDim,
                            "num_layers": numLayers,
                            "bias": True,
                            "batch_first": True,
                            "dropout": dropout if numLayers > 1 else 0.0,
                            "bidirectional": bidirectional,
                        },
                    },
                ],
            }

        elif architectureType == "Conv1d":
            # Conv1d architecture
            numConvLayers = trial.suggest_int("numConvLayers", 1, 3)
            kernelSize = trial.suggest_categorical("kernelSize", [3, 5, 7])
            stride = 1 #FIXED PARAM trial.suggest_categorical("strideConv1D", [1])  # FIX: Use stride=1 to avoid sequence collapse
            activationFunction = "GELU" # FIXED VALUE
            # trial.suggest_categorical(
            #     "encoderActivation", ["ReLU", "LeakyReLU", "GELU"]
            # )

            # Start with input channels
            inputChannels = datasetInfo["timeSeriesShape"][0]
            
            # Calculate current sequence length to avoid over-pooling
            currentSeqLen = datasetInfo["timeSeriesShape"][1]

            # Build conv layers
            layerList = []
            currentChannels = inputChannels
            for i in range(numConvLayers):
                outChannels = trial.suggest_int(f"convChannels_{i}", 32, 256)
                layerList.append(
                    {
                        "name": "Conv1d",
                        "params": {
                            "in_channels": currentChannels,
                            "out_channels": outChannels,
                            "kernel_size": kernelSize,
                            "stride": stride,
                            "padding": kernelSize // 2,
                            "bias": True,
                        },
                    }
                )
                
                # Update sequence length after conv (with padding, length stays same if stride=1)
                currentSeqLen = currentSeqLen // stride
                
                # Only add pooling if sequence is long enough (>4 after pooling)
                # This prevents sequence collapse to 0
                if currentSeqLen > 8:  # Safe threshold: allows at least 4 after pooling
                    poolType = trial.suggest_categorical(f"poolType_{i}", ["max", "avg", "none"])
                    if poolType == "max":
                        layerList.append(
                            {
                                "name": "MaxPool1d",
                                "params": {
                                    "kernel_size": 2,
                                    "stride": 2,
                                },
                            }
                        )
                        currentSeqLen = currentSeqLen // 2
                    elif poolType == "avg":
                        layerList.append(
                            {
                                "name": "AvgPool1d",
                                "params": {
                                    "kernel_size": 2,
                                    "stride": 2,
                                },
                            }
                        )
                        currentSeqLen = currentSeqLen // 2
                    # else: poolType == "none", no pooling added
                
                currentChannels = outChannels

            # Add adaptive pooling and flatten
            layerList.append(
                {
                    "name": "AdaptiveAvgPool1d",
                    "params": {
                        "output_size": 1,
                    },
                }
            )
            layerList.append(
                {
                    "name": "Flatten",
                    "params": {
                        "start_dim": 1,
                        "end_dim": -1,
                    },
                }
            )

            timeSeriesEncoderParams = {
                "activation_function": activationFunction,
                "layer_type": layerList,
            }

        elif architectureType == "MultiScaleCNN":
            # Multi-scale CNN architecture (Inception-style)
            numMultiScaleLayers = trial.suggest_int("numMultiScaleLayers", 1, 3)
            useDilation = True # FIXED PARAM trial.suggest_categorical("useDilation", [True, False])
            kernelSizesProfiles = [(3, 5, 7), (3, 7, 11), (5, 9, 13)]
            kernelSizesIndex = trial.suggest_int(
                "kernelSizesIndex", 0, len(kernelSizesProfiles) - 1
            )
            kernelSizes = kernelSizesProfiles[kernelSizesIndex]
            activationFunction = "GELU" # FIXED VALUE 
            # trial.suggest_categorical(
            #     "encoderActivation", ["ReLU", "LeakyReLU", "GELU"]
            # )

            # Start with input channels
            inputChannels = datasetInfo["timeSeriesShape"][0]

            # Build multi-scale layers
            layerList = []
            currentChannels = inputChannels
            for i in range(numMultiScaleLayers):
                branchChannels = trial.suggest_int(f"branchChannels_{i}", 32, 128)
                poolingType = "max" # FIXED VALUE
                # trial.suggest_categorical(
                #     f"poolingType_{i}", ["max", "avg", "none"]
                # )

                layerList.append(
                    {
                        "name": "MultiScaleCNN",
                        "params": {
                            "in_channels": currentChannels,
                            "branch_channels": branchChannels,
                            "kernel_sizes": kernelSizes,
                            "use_dilation": useDilation,
                            "pooling_type": poolingType,
                        },
                    }
                )

                # Update current channels: output is branch_channels * num_branches
                currentChannels = branchChannels * len(kernelSizes)

            # Add adaptive pooling and flatten
            layerList.append(
                {
                    "name": "AdaptiveAvgPool1d",
                    "params": {
                        "output_size": 1,
                    },
                }
            )
            layerList.append(
                {
                    "name": "Flatten",
                    "params": {
                        "start_dim": 1,
                        "end_dim": -1,
                    },
                }
            )

            timeSeriesEncoderParams = {
                "activation_function": activationFunction,
                "layer_type": layerList,
            }

        architectureParameters["EncoderParams"] = timeSeriesEncoderParams
        return architectureParameters

    def setUpFeedForwardHead(
        self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict
    ):
        """
        Configure the feedforward classification head.

        The feedforward head takes the concatenated output from both encoders
        (time series + global features) and produces class predictions.

        The architecture consists of:
        - Multiple fully connected layers with configurable dimensions
        - Dropout for regularization
        - Activation functions (ReLU/LeakyReLU/GELU)
        - Final output layer (added by Direct/Autoencoder class)

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter suggestions
            architectureParameters (dict): Architecture configuration containing:
                - GlobalFFEncoderParams: Global encoder config (to get embedding dim)
                - EncoderParams: Time series encoder config (to calculate output size)
            datasetInfo (dict): Dataset metadata (may be used for dimension calculations)

        Returns:
            dict: Updated architectureParameters with added key:
                - FeedForwardParams: Configuration for classification head

        Note:
            The final output layer (num_classes) is automatically added by the
            Direct or LightningAutoencoder class based on OutputDim parameter.
        """
        globalEmbeddingDim = architectureParameters["GlobalFFEncoderParams"][
            "layer_type"
        ][-1]["params"]["out_features"]

        # Calculate encoder output size based on architecture type
        encoderParams = architectureParameters["EncoderParams"]
        firstLayer = encoderParams["layer_type"][0]

        if firstLayer["name"] in ["LSTM", "GRU", "RNN"]:
            # RNN architecture
            hiddenDim = firstLayer["params"]["hidden_size"]
            bidirectional = firstLayer["params"]["bidirectional"]
            rnnOutputSize = hiddenDim * (2 if bidirectional else 1)
            encoderOutputSize = rnnOutputSize

        elif firstLayer["name"] == "Conv1d":
            # Conv1d architecture - find the last conv layer before pooling
            lastConvLayer = None
            for layer in encoderParams["layer_type"]:
                if layer["name"] == "Conv1d":
                    lastConvLayer = layer

            if lastConvLayer:
                # After AdaptiveAvgPool1d(1) and Flatten, output size = out_channels
                encoderOutputSize = lastConvLayer["params"]["out_channels"]
            else:
                encoderOutputSize = 64  # Fallback

        elif firstLayer["name"] == "MultiScaleCNN":
            # MultiScaleCNN architecture - find the last MultiScaleCNN layer before pooling
            lastMultiScaleLayer = None
            for layer in encoderParams["layer_type"]:
                if layer["name"] == "MultiScaleCNN":
                    lastMultiScaleLayer = layer

            if lastMultiScaleLayer:
                # After AdaptiveAvgPool1d(1) and Flatten, output size = branch_channels * num_branches
                branchChannels = lastMultiScaleLayer["params"]["branch_channels"]
                numBranches = len(lastMultiScaleLayer["params"]["kernel_sizes"])
                encoderOutputSize = branchChannels * numBranches
            else:
                encoderOutputSize = 64  # Fallback
        else:
            # Fallback
            encoderOutputSize = 64

        combinedInputSize = encoderOutputSize + globalEmbeddingDim

        # Suggest feedforward head architecture
        numHiddenLayers = trial.suggest_int("numFFLayers", 1, 4)
        # Coefficient to determine first layer size relative to input
        # e.g., 0.5 means first layer has half the neurons of input
        ffSizeCoefficient = trial.suggest_float("ffSizeCoefficient", 0.3, 0.8)
        ffStartDim = max(32, int(combinedInputSize * ffSizeCoefficient))  # First layer size
        ffDropout = trial.suggest_float("ffDropout", 0.0, 0.4)
        ffActivation = "LeakyReLU"  # FIXED VALUE
        # trial.suggest_categorical(
        #     "ffActivation", ["ReLU", "LeakyReLU", "GELU"]
        # )

        # Build layer list with linearly decreasing dimensions
        # Scale from ffStartDim down to 32 (minimum before output layer) across numHiddenLayers
        layerList_clean = []
        
        minFinalDim = 32  # Minimum dimension before output layer
        
        if numHiddenLayers == 1:
            # Single layer: use ffStartDim directly
            dimensions = [combinedInputSize, ffStartDim]
        else:
            # Multiple layers: linearly interpolate from ffStartDim down to minFinalDim
            # Example: 3 layers, ffStartDim=200 -> [combinedInput, 200, 116, 32]
            step = (ffStartDim - minFinalDim) / (numHiddenLayers - 1)
            dimensions = [combinedInputSize] + [max(minFinalDim, int(ffStartDim - step * i)) for i in range(numHiddenLayers)]
        
        for i in range(numHiddenLayers):
            layerList_clean.append(
                {
                    "name": "Linear",
                    "params": {
                        "in_features": dimensions[i],
                        "out_features": dimensions[i + 1],
                        "bias": True,
                    },
                }
            )
            # Add dropout BEFORE the next layer (not after the last one)
            if ffDropout > 0 and i < numHiddenLayers - 1:
                layerList_clean.append(
                    {
                        "name": "Dropout",
                        "params": {
                            "p": ffDropout,
                            "inplace": False,
                        },
                    }
                )

        # Note: Final output layer will be added by Direct/Autoencoder class
        # The last layer MUST have "out_features" for Direct to append the output layer
        feedForwardParams = {
            "activation_function": ffActivation,
            "layer_type": layerList_clean,
        }

        architectureParameters["FeedForwardParams"] = feedForwardParams
        return architectureParameters

    def setUpDecoder(
        self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict
    ):
        """
        Configure decoder architecture for autoencoder models.

        The decoder mirrors the encoder architecture to reconstruct input data.
        This is only used when MacroArchitecture="Autoencoder".

        Decoder architectures by encoder type:
        - Recurrent: Uses teacher forcing during training, autoregressive during inference
          * Encoder hidden state becomes decoder's initial hidden state
          * Reconstructs original time series sequence
        - Conv1d: Uses transposed convolutions for upsampling
          * Symmetrically reverses encoder convolutions
          * Includes adaptive pooling for exact length matching
        - MultiScaleCNN: Uses transposed convolutions to reverse multi-scale features
          * Mirrors multi-scale architecture in reverse
          * Reconstructs to original input dimensions

        Also sets up global feature decoder (mirrors global encoder).

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter suggestions
            architectureParameters (dict): Architecture configuration containing:
                - EncoderParams: Time series encoder config to mirror
                - GlobalFFEncoderParams: Global encoder config to mirror
            datasetInfo (dict): Dataset metadata containing:
                - timeSeriesShape: Target reconstruction shape (channels, length)
                - globalFeaturesShape: Target global features shape

        Returns:
            dict: Updated architectureParameters with added keys:
                - DecoderParams: Configuration for time series decoder
                - GlobalFFDecoderParams: Configuration for global feature decoder

        Note:
            For RNN decoders, the architecture supports teacher forcing during
            training for stable learning, and autoregressive generation during inference.
        """
        encoderParams = architectureParameters["EncoderParams"]
        firstLayer = encoderParams["layer_type"][0]

        # Mirror the encoder architecture
        if firstLayer["name"] in ["LSTM", "GRU", "RNN"]:
            # RNN decoder
            rnnType = firstLayer["name"]
            hiddenDim = firstLayer["params"]["hidden_size"]
            numLayers = firstLayer["params"]["num_layers"]
            bidirectional = firstLayer["params"]["bidirectional"]
            dropout = firstLayer["params"]["dropout"]
            activationFunction = encoderParams["activation_function"]

            # Get encoder output size
            encoderOutputSize = hiddenDim * (2 if bidirectional else 1)

            # Output should reconstruct the input (34 channels)
            outputSize = datasetInfo["timeSeriesShape"][0]

            # FIXED: Proper Autoregressive RNN Decoder Architecture
            # Encoder flow:
            #   (batch, 34, 160) -> permute -> (batch, 160, 34) -> LSTM
            #   -> (batch, 160, hidden_size*directions) -> take last timestep
            #   -> (batch, hidden_size*directions)
            #
            # Decoder flow (NEW - Teacher Forcing + Autoregressive):
            #   Training: (batch, hidden_size*directions) -> use as h0 (initial hidden state)
            #             Teacher forcing input: [START, x_0, x_1, ..., x_{T-2}]
            #             -> LSTM(input, h0) -> (batch, 160, hidden_size)
            #             -> Linear projection -> (batch, 160, 34)
            #             -> permute -> (batch, 34, 160)
            #
            #   Inference: Same but with zero inputs (autoregressive from hidden state)
            #
            # Key improvements:
            # - Uses encoder output as initial hidden state (not repeated input!)
            # - Teacher forcing during training (stable learning)
            # - Autoregressive generation during inference
            # - No Python loops (single RNN forward pass)

            layerList = []

            # RNN layer: Must match encoder configuration exactly
            # IMPORTANT: Decoder bidirectional must match encoder bidirectional
            # IMPORTANT: Decoder num_layers must match encoder num_layers
            # This ensures encoder output shape matches decoder expected hidden state shape
            decoder_bidirectional = bidirectional  # Match encoder bidirectional setting

            layerList.append(
                {
                    "name": rnnType,
                    "params": {
                        "input_size": outputSize,  # 34 (for teacher forcing)
                        "hidden_size": hiddenDim,  # Keep encoder's hidden size
                        "num_layers": numLayers,  # MUST match encoder layers
                        "bias": True,
                        "batch_first": True,
                        "dropout": dropout if numLayers > 1 else 0.0,
                        "bidirectional": decoder_bidirectional,  # Match encoder
                    },
                }
            )

            # Linear projection: account for bidirectional output
            # If bidirectional, RNN outputs hidden_size * 2
            decoder_output_features = hiddenDim * (2 if decoder_bidirectional else 1)
            layerList.append(
                {
                    "name": "Linear",
                    "params": {
                        "in_features": decoder_output_features,  # hidden_size or hidden_size*2
                        "out_features": outputSize,  # 34
                        "bias": True,
                    },
                }
            )

            # Now output is (batch, seq_len, 34) which becomes (batch, 34, 160) after permute

            decoderParams = {
                "activation_function": activationFunction,
                "layer_type": layerList,
            }

        elif firstLayer["name"] == "Conv1d":
            # Conv1d decoder - symmetric architecture using ConvTranspose1d
            # The encoder ends with: Conv layers -> AdaptiveAvgPool1d(1) -> Flatten
            # Decoder: Unflatten -> ConvTranspose layers (reversed)

            # Get conv layers from encoder
            convLayers = [
                l for l in encoderParams["layer_type"] if l["name"] == "Conv1d"
            ]
            poolLayers = [
                l
                for l in encoderParams["layer_type"]
                if l["name"] in ["MaxPool1d", "AvgPool1d"]
            ]

            lastConvChannels = convLayers[-1]["params"]["out_channels"]
            kernelSize = convLayers[0]["params"]["kernel_size"]
            stride = convLayers[0]["params"]["stride"]

            # Calculate starting sequence length after encoder
            # After AdaptiveAvgPool1d(1), we have seq_len=1
            startSeqLen = 1

            # Calculate how many times we need to upsample
            numPoolLayers = len(poolLayers)

            layerList = []

            # First, unflatten from (batch, channels) to (batch, channels, 1)
            layerList.append(
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (lastConvChannels, startSeqLen),
                    },
                }
            )

            # Reverse the conv layers
            reversedConvLayers = list(reversed(convLayers))

            # Build decoder layers symmetrically
            currentChannels = lastConvChannels
            for i, convLayer in enumerate(reversedConvLayers):
                # For the last layer, output should be original input channels (34)
                if i == len(reversedConvLayers) - 1:
                    outChannels = datasetInfo["timeSeriesShape"][0]  # 34
                else:
                    # Output channels should be input channels of the corresponding encoder layer
                    outChannels = reversedConvLayers[i + 1]["params"]["out_channels"]

                # Add upsampling with ConvTranspose1d
                # Use stride=2 to upsample if there was pooling in encoder
                useStride = 2 if i < numPoolLayers else stride

                layerList.append(
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": currentChannels,
                            "out_channels": outChannels,
                            "kernel_size": kernelSize,
                            "stride": useStride,
                            "padding": kernelSize // 2,
                            "output_padding": useStride - 1 if useStride > 1 else 0,
                            "bias": True,
                        },
                    }
                )

                # Update current channels for next layer
                currentChannels = outChannels

            # Add final adjustment layer to match exact sequence length
            # Use adaptive interpolation if needed
            targetSeqLen = datasetInfo["timeSeriesShape"][1]  # 160
            layerList.append(
                {
                    "name": "AdaptiveAvgPool1d",
                    "params": {
                        "output_size": targetSeqLen,
                    },
                }
            )

            decoderParams = {
                "activation_function": encoderParams["activation_function"],
                "layer_type": layerList,
            }

        elif firstLayer["name"] == "MultiScaleCNN":
            # MultiScaleCNN decoder - symmetric architecture using ConvTranspose1d
            # The encoder ends with: MultiScaleCNN layers -> AdaptiveAvgPool1d(1) -> Flatten
            # Decoder: Unflatten -> ConvTranspose layers (reversed)

            # Get MultiScaleCNN layers from encoder
            multiScaleLayers = [
                l for l in encoderParams["layer_type"] if l["name"] == "MultiScaleCNN"
            ]
            poolLayers = [
                l
                for l in encoderParams["layer_type"]
                if l["name"] in ["MaxPool1d", "AvgPool1d"]
            ]

            # Get the last MultiScaleCNN layer's output channels
            lastLayer = multiScaleLayers[-1]
            branchChannels = lastLayer["params"]["branch_channels"]
            numBranches = len(lastLayer["params"]["kernel_sizes"])
            lastMultiScaleChannels = branchChannels * numBranches

            # Calculate starting sequence length after encoder
            # After AdaptiveAvgPool1d(1), we have seq_len=1
            startSeqLen = 1

            # Calculate how many times we need to upsample
            numPoolLayers = len(poolLayers)

            layerList = []

            # First, unflatten from (batch, channels) to (batch, channels, 1)
            layerList.append(
                {
                    "name": "Unflatten",
                    "params": {
                        "dim": 1,
                        "unflattened_size": (lastMultiScaleChannels, startSeqLen),
                    },
                }
            )

            # Reverse the MultiScaleCNN layers and create ConvTranspose1d layers
            reversedLayers = list(reversed(multiScaleLayers))

            # Build decoder layers symmetrically
            currentChannels = lastMultiScaleChannels
            for i, msLayer in enumerate(reversedLayers):
                # Calculate the output channels for this layer
                if i == len(reversedLayers) - 1:
                    # Last layer should output original input channels (34)
                    outChannels = datasetInfo["timeSeriesShape"][0]
                else:
                    # Output channels should match the input of the corresponding encoder layer
                    nextLayer = reversedLayers[i + 1]
                    outChannels = nextLayer["params"]["branch_channels"] * len(
                        nextLayer["params"]["kernel_sizes"]
                    )

                # Use ConvTranspose1d for upsampling
                # Use stride=2 to upsample if there was pooling in encoder
                useStride = 2 if i < numPoolLayers else 1

                # Use a moderate kernel size for transposed convolution
                kernelSize = 5  # Good balance for upsampling

                layerList.append(
                    {
                        "name": "ConvTranspose1d",
                        "params": {
                            "in_channels": currentChannels,
                            "out_channels": outChannels,
                            "kernel_size": kernelSize,
                            "stride": useStride,
                            "padding": kernelSize // 2,
                            "output_padding": useStride - 1 if useStride > 1 else 0,
                            "bias": True,
                        },
                    }
                )

                # Update current channels for next layer
                currentChannels = outChannels

            # Add final adjustment layer to match exact sequence length
            targetSeqLen = datasetInfo["timeSeriesShape"][1]  # 160
            layerList.append(
                {
                    "name": "AdaptiveAvgPool1d",
                    "params": {
                        "output_size": targetSeqLen,
                    },
                }
            )

            decoderParams = {
                "activation_function": encoderParams["activation_function"],
                "layer_type": layerList,
            }

        else:
            # Fallback
            decoderParams = encoderParams

        # Mirror global decoder - reverse the encoder architecture
        globalEncoderParams = architectureParameters["GlobalFFEncoderParams"]
        globalEncoderLayers = globalEncoderParams["layer_type"]

        # Get only Linear layers from encoder (skip Dropout)
        globalEncoderLinearLayers = [
            l for l in globalEncoderLayers if l["name"] == "Linear"
        ]

        # Reverse the architecture
        globalDecoderLayerList = []
        for i, encoderLayer in enumerate(reversed(globalEncoderLinearLayers)):
            # Swap in_features and out_features
            inFeatures = encoderLayer["params"]["out_features"]
            outFeatures = encoderLayer["params"]["in_features"]

            globalDecoderLayerList.append(
                {
                    "name": "Linear",
                    "params": {
                        "in_features": inFeatures,
                        "out_features": outFeatures,
                        "bias": True,
                    },
                }
            )

            # Add dropout between layers (not after last)
            if i < len(globalEncoderLinearLayers) - 1:
                # Find dropout from encoder if it exists
                for encLayer in globalEncoderLayers:
                    if encLayer["name"] == "Dropout":
                        globalDecoderLayerList.append(
                            {"name": "Dropout", "params": encLayer["params"]}
                        )
                        break

        globalDecoderParams = {
            "activation_function": globalEncoderParams["activation_function"],
            "layer_type": globalDecoderLayerList,
        }

        architectureParameters["DecoderParams"] = decoderParams
        architectureParameters["GlobalFFDecoderParams"] = globalDecoderParams
        return architectureParameters

    def apply_he_initialization(self, model, activation_type="ReLU"):
        """
        Apply He (Kaiming) initialization to model weights.

        He initialization sets initial weights based on the size of the previous layer
        to maintain variance across layers. This is especially important for deep
        networks and helps with gradient flow during training.

        Applied to:
        - Linear layers: Uses kaiming_normal_ with appropriate nonlinearity
        - Conv1d layers: Uses kaiming_normal_ with appropriate nonlinearity
        - RNN layers (LSTM/GRU):
          * Input-hidden weights: He initialization
          * Hidden-hidden weights: Orthogonal initialization (better for recurrence)
          * Biases: Small constant (0.01)
          * LSTM forget gate bias: Initialized to 1.0 for better gradient flow

        Args:
            model (torch.nn.Module): PyTorch model to initialize
            activation_type (str): Type of activation function used in the model.
                                  Options: "ReLU", "LeakyReLU", "GELU"
                                  Affects the initialization mode.

        Note:
            - He initialization is optimal for ReLU-like activations
            - GELU also benefits from this initialization strategy
            - Biases are initialized to small positive values (0.01)
            - LSTM forget gates use bias=1.0 to prevent early gradient vanishing
        """
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # He initialization for Linear layers
                if activation_type == "LeakyReLU":
                    # For LeakyReLU, specify the negative slope (default 0.01)
                    torch.nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
                    )
                elif activation_type in ["ReLU", "GELU"]:
                    # For ReLU and GELU
                    torch.nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )

                # Initialize bias to small constant
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)

            elif isinstance(module, torch.nn.Conv1d):
                # He initialization for Conv1d layers
                if activation_type == "LeakyReLU":
                    torch.nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
                    )
                elif activation_type in ["ReLU", "GELU"]:
                    torch.nn.init.kaiming_normal_(
                        module.weight, mode="fan_in", nonlinearity="relu"
                    )

                # Initialize bias to small constant
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)

            elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
                # For RNN layers, initialize with orthogonal initialization (common practice)
                for param_name, param in module.named_parameters():
                    if "weight_ih" in param_name:
                        # Input-hidden weights: use He initialization
                        torch.nn.init.kaiming_normal_(
                            param.data, mode="fan_in", nonlinearity="relu"
                        )
                    elif "weight_hh" in param_name:
                        # Hidden-hidden weights: use orthogonal initialization for stability
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in param_name:
                        # Initialize biases to small constant
                        torch.nn.init.constant_(param.data, 0.01)
                        # For LSTM, set forget gate bias to 1 (helps with gradient flow)
                        if isinstance(module, torch.nn.LSTM):
                            n = param.data.size(0)
                            # Forget gate bias
                            param.data[n // 4 : n // 2].fill_(1.0)

    def objective_kfold(self, trial: optuna.trial.Trial) -> float:
        """
        K-Fold Cross-Validation objective function for Optuna optimization.

        This is the core optimization objective that:
        1. Suggests hyperparameters (architecture, learning rate, regularization, etc.)
        2. Creates a data loader with optional windowing
        3. Builds encoder, decoder (if autoencoder), and feedforward head
        4. Trains models on each fold with early stopping and checkpointing
        5. Returns mean F1 score across all folds

        Hyperparameters optimized:
        - MacroArchitecture: Direct classification vs Autoencoder
        - Architecture type: Recurrent/Conv1d/MultiScaleCNN
        - Architecture-specific parameters (hidden dims, layers, etc.)
        - Windowing: window_size, stride
        - Learning rate and regularization
        - Learning rate scheduler type and parameters
        - Dropout rates and activation functions

        Args:
            trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters

        Returns:
            float: Mean validation F1 score across all K folds

        Raises:
            optuna.TrialPruned: If any fold fails or if the trial should be pruned
                               based on MedianPruner criteria

        Side Effects:
            - Creates DataModule and loads data
            - Trains multiple models (one per fold)
            - Saves checkpoints to disk
            - Stores fold_scores, mean_f1, std_f1 as trial user attributes
            - Prints fold progress (suppressed via enable_progress_bar=False)

        Note:
            For autoencoder architecture, test data is included in training for
            unsupervised reconstruction learning.
        """
        # ==================== STEP 1: Suggest Macro Architecture ====================
        # Choose between direct classification or autoencoder-based approach
        macroArchitecture = "Autoencoder" # FIXED PARAM
        
        # trial.suggest_categorical(
        #     "MacroArchitecture", ["Direct", "Autoencoder"]
        # )

        # ==================== STEP 2: Configure Data Windowing ====================
        # Windowing splits time series into smaller overlapping segments (MODEL-LEVEL, not data-loader level)
        # This can help the model learn from more samples and capture local patterns
        use_windowing = True # FIXED PARAM trial.suggest_categorical("use_windowing", [True, False])

        if use_windowing:
            # Window size: how much of the sequence to process at once
            window_size = trial.suggest_int("window_size", 5, 40)

            # Stride: step size between windows (lower = more overlap)
            # Use categorical to prefer common overlap patterns
            stride_ratio = 1.0  # FIXED VALUE
            # trial.suggest_categorical(
            #     "stride_ratio", [0.25, 0.5, 0.75, 1.0]
            # )
            stride = int(window_size * stride_ratio)

            # Aggregation method: how to combine window predictions
            aggregation_method = "majority_vote"  # FIXED VALUE
            # trial.suggest_categorical(
            #     "aggregation_method",
            #     ["avg_probs", "majority_vote"],
            # ) # REDUCED OPTIONS

            # Window loss weight: auxiliary supervision on individual windows
            # 0 = only sample-level loss, >0 = also supervise individual windows
            window_loss_weight = 0.2 # FIXED VALUE
        else:
            window_size = 160
            stride = 160

        # ==================== STEP 2.5: Configure Data Augmentation ====================
        # Data augmentation helps improve generalization by creating variations of training data
        use_augmentation = True
        
        augmentation_config = None
        if use_augmentation:
            # Jittering: Add Gaussian noise to time series
            jitter_strength = trial.suggest_float("jitter_strength", 0.01, 0.1, log=True)
            
            # Scaling: Random amplitude scaling
            scaling_range = trial.suggest_float("scaling_range", 0.08, 0.15)
            
            # Time Warping: Smooth time axis distortion
            time_warp_strength = trial.suggest_float("time_warp_strength", 0.01, 0.25)
            
            # Create config in the correct format for AugmentationPipeline.from_config()
            augmentation_config = {
                "jitter_enabled": True,
                "jitter_strength": jitter_strength,
                "scaling_enabled": True,
                "scaling_range": scaling_range,
                "time_warp_enabled": True,
                "time_warp_strength": time_warp_strength
            }

        # ==================== STEP 3: Setup Data Loading ====================
        # For autoencoders, include test data in training (unsupervised reconstruction)
        # For direct classification, only use labeled training data
        includeTestInTrain = macroArchitecture == "Autoencoder"

        # Create K-Fold data loader with windowing parameters
        kfold_data_params = self.data_params.copy()
        kfold_data_params["use_windowing"] = (
            False  # ← Always False! Windowing is at model level
        )
        kfold_data_params["window_size"] = 160  # Full sequence length
        kfold_data_params["stride"] = 160  # No overlap in DataLoader
        n_folds = kfold_data_params["n_folds"]
        
        # Add augmentation configuration to data params
        kfold_data_params["augmentation_config"] = augmentation_config
        kfold_data_params["augmentation_seed"] = trial.number * 1000  # Reproducible per trial

        kfold_dataLoader = DataModule(params=kfold_data_params)
        # Load training data and prepare for K-fold splits
        # If includeTestInTrain=True (autoencoder), test data will be loaded WITH augmentation
        kfold_dataLoader.setup(stage="fit", includeTestInTrain=includeTestInTrain)
        datasetInfo = kfold_dataLoader.getDatasetInfo()
        
        # CRITICAL FIX: When using WindowedModelWrapper, the base model receives windowed sequences
        # So we need to adjust datasetInfo to reflect what the base model actually sees
        if use_windowing:
            datasetInfo["timeSeriesShape"] = (datasetInfo["timeSeriesShape"][0], window_size)
            datasetInfo["use_windowing"] = True
            datasetInfo["window_size"] = window_size
            datasetInfo["stride"] = stride

        # ==================== STEP 4: Build Architecture Configuration ====================
        # Architecture parameters will be the same across all folds in this trial
        archParams = {}

        # Setup encoders (both time series and global feature encoders)
        archParams = self.setUpEncoder(trial, archParams, datasetInfo)

        # Setup feedforward classification head
        archParams = self.setUpFeedForwardHead(trial, archParams, datasetInfo)

        # For autoencoder: add decoder and reconstruction loss weight
        if macroArchitecture == "Autoencoder":
            archParams = self.setUpDecoder(trial, archParams, datasetInfo)
            archParams["ReconstructionLossWeight"] = trial.suggest_float(
                "ReconstructionLossWeight", 1e-2, 0.5, log=True
            )

        # ==================== STEP 5: Configure Training Parameters ====================
        # Common parameters for all architectures
        archParams["OutputDim"] = 3  # Number of classes (pain levels)
        archParams["LearningRate"] = 0.0005 # FIXED VALUE
        # trial.suggest_float(
        #     "LearningRate", 1e-5, 1e-2, log=True
        # )
        archParams["RegularizationWeight"] = trial.suggest_float(
            "RegularizationWeight", 1e-3, 1e1, log=True
        )

        useClassWeights = trial.suggest_categorical("useClassWeights", [True, False])
        archParams["ClassWeightsPath"] = self.data_params.get(
            "class_weights_path", "../dataset/PirateProcessed/class_weights.yaml" if useClassWeights else None
        )

        # Maximum training epochs (early stopping may terminate earlier)
        max_epochs = 100

        # ==================== STEP 6: Configure Learning Rate Scheduler ====================
        # Different schedulers for adaptive learning rate adjustment
        scheduler_type = "ReduceLROnPlateau"  # FIXED VALUE
        # trial.suggest_categorical(
        #     "SchedulerType",
        #     ["ReduceLROnPlateau", "CosineAnnealing", "CosineAnnealingWarmRestarts"],
        # )
        archParams["SchedulerType"] = scheduler_type

        if scheduler_type == "ReduceLROnPlateau":
            archParams["Patience"] = 10 # FIXED VALUE
                # trial.suggest_int(
                #     "SchedulerPatience", 3, 10
                # )  # Patience for LR reduction
            archParams["SchedulerFactor"] = 0.1  # FIXED VALUE
            # trial.suggest_float(
            #     "SchedulerFactor", 0.1, 0.25
            # )
            archParams["SchedulerMinLR"] = 0  # FIXED VALUE
            # trial.suggest_float(
            #     "SchedulerMinLR", 1e-6, 1e-4, log=True
            # )
        elif scheduler_type == "CosineAnnealing":
            # Cosine annealing: smooth LR decay following cosine curve
            # Full cycle matches training duration
            archParams["T_max"] = max_epochs
            archParams["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            # Cosine annealing with periodic restarts (helps escape local minima)
            archParams["T_0"] = trial.suggest_int(
                "T_0", 5, 20
            )  # Initial restart period
            archParams["T_mult"] = trial.suggest_int(
                "T_mult", 1, 3
            )  # Period multiplier after restart
            archParams["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)

        # Early stopping: stops training if no improvement after patience epochs
        early_stopping_patience =  15  # FIXED VALUE

        # ==================== STEP 7: K-Fold Cross-Validation Training ====================
        # Track validation scores across all folds
        fold_scores = []

        # Train a separate model on each fold to get robust performance estimate
        for fold_idx in range(n_folds):
            # Set deterministic seed for this fold (trial-specific + fold-specific)
            # This ensures reproducibility but allows variation across folds
            fold_seed = trial.number * 1000 + fold_idx
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(fold_seed)
                torch.cuda.manual_seed_all(fold_seed)
            
            # Prepare data for this specific fold (different train/val split)
            kfold_dataLoader.setup_fold(
                fold_idx, include_test_in_train=includeTestInTrain
            )
            trainLoader = kfold_dataLoader.train_dataloader()
            valLoader = kfold_dataLoader.val_dataloader()
            
            # Log fold statistics to debug why fold 2 performs consistently better
            # Get validation labels to check class distribution (using torch operations)
            val_labels_list = []
            for batch in valLoader:
                if isinstance(batch, dict):
                    val_labels_list.append(batch['label'])
                else:
                    val_labels_list.append(batch[1])
            val_labels = torch.cat(val_labels_list, dim=0)
            val_label_counts = torch.bincount(val_labels, minlength=3)
            print(f"[FOLD STATS] Fold {fold_idx}: Val set size={len(val_labels)}, "
                  f"Class distribution: no_pain={val_label_counts[0].item()}, "
                  f"low_pain={val_label_counts[1].item()}, high_pain={val_label_counts[2].item()}")

            # Create a fresh model instance for this fold (no weight sharing between folds)
            # CRITICAL: Deep copy archParams to prevent mutation across folds
            # The Direct/Autoencoder classes modify the params dict (e.g., appending output layer)
            archParams_copy = copy.deepcopy(archParams)
            
            # Debug: Log architecture params before model creation to check for mutation
            print(f"[DEBUG] Fold {fold_idx}: FeedForward layers before model creation: "
                  f"{archParams_copy['FeedForwardParams']['layer_type']}")

            if macroArchitecture == "Direct":
                base_model = Direct(archParams_copy)
            else:
                base_model = LightningAutoencoder(archParams_copy)
            
            # Debug: Check if archParams_copy was mutated after model creation
            # If this differs from the "before" log, we have architecture mutation
            if archParams_copy['FeedForwardParams']['layer_type'] != archParams['FeedForwardParams']['layer_type']:
                print(f"[DEBUG] WARNING: Fold {fold_idx}: Architecture was mutated! "
                      f"After model creation: {archParams_copy['FeedForwardParams']['layer_type']}")

            # Apply He initialization to base model
            ff_activation = archParams["FeedForwardParams"]["activation_function"]
            self.apply_he_initialization(base_model, activation_type=ff_activation)

            # Wrap with WindowedModelWrapper if windowing is enabled
            if use_windowing:
                model = WindowedModelWrapper(
                    base_model=base_model,
                    window_size=window_size,
                    stride=stride,
                    aggregation_method=aggregation_method,
                    window_loss_weight=window_loss_weight,
                )
            else:
                model = base_model

            # Add early stopping callback
            early_stopping_callback = EarlyStopping(
                monitor="val_F1",
                patience=early_stopping_patience,
                mode="max",
                verbose=False,
            )

            # Tensorboard Logging
            # CRITICAL: version=None disables automatic version numbering
            # This ensures each trial+fold combination uses the same directory,
            # preventing checkpoint confusion from multiple runs
            logger = TensorBoardLogger(
                save_dir=f"FinalPipelineLogs/Study_{trial.study.study_name}",
                name=f"{trial.study.study_name}_trial_{trial.number}_fold_{fold_idx}",
                version=None,  # Disable version numbering to prevent directory proliferation
                log_graph=True,
            )

            # Model checkpointing: saves best model based on validation F1
            # Save checkpoints in the same directory as TensorBoard logs
            checkpoint_callback = ModelCheckpoint(
                dirpath=logger.log_dir + "/checkpoints",  # Save in same dir as logs
                monitor="val_F1",
                mode="max",
                save_top_k=1,
                filename=f"trial-{trial.number}-fold-{fold_idx}-"
                + "{epoch:02d}-{val_F1:.3f}",
                verbose=False,
            )

            # Optional: Stochastic Weight Averaging (SWA) for improved generalization
            stochastic_weight_avg_callback = StochasticWeightAveraging(
                swa_lrs=archParams["LearningRate"]
                )

            # Create PyTorch Lightning trainer with configured callbacks
            trainer = Trainer(
                max_epochs=max_epochs,
                enable_progress_bar=True,
                enable_model_summary=False,
                callbacks=[early_stopping_callback, checkpoint_callback, stochastic_weight_avg_callback],
                enable_checkpointing=True,
                logger=logger,
                log_every_n_steps=20,
                gradient_clip_val=1.0,
            )

            # Train the model
            try:
                trainer.fit(model, trainLoader, valLoader)

                # Get best validation F1 for this fold from checkpoint callback
                best_f1 = (
                    checkpoint_callback.best_model_score.item()
                    if checkpoint_callback.best_model_score is not None
                    else 0.0
                )
                fold_scores.append(best_f1)
                trial.report(np.mean(fold_scores), step=fold_idx)

                # Log the trial progress
                trial.set_user_attr(f"fold_{fold_idx}", best_f1)


                # Check if this trial should be pruned (stopped early)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Validate architecture consistency after training
                if checkpoint_callback.best_model_path:
                    # Extract expected architecture from trial parameters
                    expected_ff_layers = trial.params.get("numFFLayers", None)

                    # Get actual architecture from checkpoint
                    actual_fingerprint = self._get_architecture_fingerprint(
                        checkpoint_callback.best_model_path
                    )
                    actual_ff_layers = actual_fingerprint.get(
                        "feedforward_layers", None
                    )

                    # Validate architecture matches expectations
                    if expected_ff_layers is not None and actual_ff_layers is not None:
                        if actual_ff_layers + 1 != expected_ff_layers:
                            print(
                                f"⚠️ WARNING: Architecture mismatch detected after training fold {fold_idx}!"
                            )
                            print(f"   Expected FF layers: {expected_ff_layers}")
                            print(f"   Actual FF layers: {actual_ff_layers}")
                            print(
                                f"   Checkpoint: {checkpoint_callback.best_model_path}"
                            )
                            print(
                                f"   This suggests the checkpoint may be from a previous trial/run!"
                            )
                        else:
                            print(
                                f"✓ Fold {fold_idx} architecture validated: {expected_ff_layers} FF layers"
                            )

            except Exception as e:
                if isinstance(e, optuna.TrialPruned):
                    print(f"Trial pruned during fold {fold_idx}.")
                    raise optuna.TrialPruned() from e

                print(f"Fold {fold_idx} failed with error: {e}")
                trial.set_user_attr("Error", str(e))
                # Prune the trial if any fold fails
                raise optuna.TrialPruned() from e

        # ==================== STEP 8: Calculate Cross-Validation Metrics ====================
        # Aggregate performance across all folds
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)

        # Store detailed results for later analysis
        # Convert to compact string format to avoid database size limits
        try:
            fold_scores_str = ",".join([f"{score:.6f}" for score in fold_scores])
            # Individual fold scores as CSV string
            trial.set_user_attr("mean_f1", float(mean_f1))  # Mean across folds
            # Standard deviation (stability measure)
            trial.set_user_attr("std_f1", float(std_f1))
        except Exception as e:
            # If database fails, log warning but continue (metrics are still returned)
            print(f"⚠️ Warning: Could not store user attributes in database: {e}")


        # Return mean F1 as the optimization objective
        return mean_f1

    def _retrain_best_model(
        self, trial_wrapper, archParams, macroArch, data_loader, best_params
    ):
        """
        Retrain the best model from scratch using Optuna's best hyperparameters.

        This is a fallback method when checkpoint files are not available.
        It trains the model on ALL folds' training data combined for maximum performance.

        Args:
            trial_wrapper: Mock trial object with best parameters
            archParams (dict): Architecture configuration
            macroArch (str): "Direct" or "Autoencoder"
            data_loader (DataModule): Data loader instance
            best_params (dict): Best hyperparameters from Optuna

        Returns:
            torch.nn.Module: Trained base model (before windowing wrapper)
        """
        print("Retraining model on full dataset...")

        # Create fresh model instance
        # CRITICAL: Deep copy archParams to prevent mutation
        archParams_copy = copy.deepcopy(archParams)
        if macroArch == "Direct":
            base_model = Direct(params=archParams_copy)
        else:
            base_model = LightningAutoencoder(params=archParams_copy)

        # Apply He initialization
        ff_activation = archParams["FeedForwardParams"]["activation_function"]
        self.apply_he_initialization(base_model, activation_type=ff_activation)

        # Wrap with WindowedModelWrapper if windowing was used during optimization
        use_windowing = best_params.get("use_windowing", False)
        if use_windowing:
            print("Wrapping with WindowedModelWrapper for training...")
            window_size = best_params.get("window_size", 160)
            stride_ratio = best_params.get("stride_ratio", 1.0)
            stride = int(window_size * stride_ratio)
            aggregation_method = best_params.get("aggregation_method", "avg_probs")
            window_loss_weight = 0.2

            training_model = WindowedModelWrapper(
                base_model=base_model,
                window_size=window_size,
                stride=stride,
                aggregation_method=aggregation_method,
                window_loss_weight=window_loss_weight,
            )
        else:
            training_model = base_model

        # Prepare combined training data (train all folds together for best performance)
        includeTestInTrain = macroArch == "Autoencoder"
        # First load test data if we need it for autoencoder training
        if includeTestInTrain:
            data_loader.setup(stage="test", includeTestInTrain=False)
        # Then load training data
        data_loader.setup(stage="fit", includeTestInTrain=includeTestInTrain)
        train_loader = data_loader.train_dataloader()
        val_loader = data_loader.val_dataloader()

        # Get training parameters
        max_epochs = 100
        early_stopping_patience = best_params.get("EarlyStoppingPatience", 15)

        # Setup callbacks
        early_stopping_callback = EarlyStopping(
            monitor="val_F1",
            patience=early_stopping_patience,
            mode="max",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_F1",
            mode="max",
            save_top_k=1,
            filename="best-retrained-{epoch:02d}-{val_F1:.3f}",
            verbose=True,
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=[early_stopping_callback, checkpoint_callback],
            enable_checkpointing=True,
        )

        print(f"Training for up to {max_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        if use_windowing:
            print(
                f"Using windowing: window_size={window_size}, stride={stride}, aggregation={aggregation_method}"
            )
        print("-" * 60)

        # Train the model (with or without windowing wrapper)
        trainer.fit(training_model, train_loader, val_loader)

        print("-" * 60)
        print(f"✓ Retraining completed!")
        print(f"Best validation F1: {checkpoint_callback.best_model_score:.4f}")

        # Load best checkpoint from retraining
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            print(f"Loading best checkpoint: {best_checkpoint}")
            if use_windowing:
                # Load the wrapped model and extract base model
                wrapped_model = WindowedModelWrapper.load_from_checkpoint(
                    best_checkpoint,
                    base_model=base_model,
                    window_size=window_size,
                    stride=stride,
                    aggregation_method=aggregation_method,
                    window_loss_weight=window_loss_weight,
                )
                # Return the unwrapped base model
                base_model = wrapped_model.base_model
            else:
                # Load the base model directly
                if macroArch == "Direct":
                    base_model = Direct.load_from_checkpoint(
                        best_checkpoint, params=archParams
                    )
                else:
                    base_model = LightningAutoencoder.load_from_checkpoint(
                        best_checkpoint, params=archParams
                    )

        return base_model

    def _get_architecture_fingerprint(self, checkpoint_path: str) -> dict:
        """
        Extract architecture fingerprint from checkpoint to verify consistency.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            dict: Architecture fingerprint with layer counts and types
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", {})

            # Count layers by type
            encoder_layers = len(
                [k for k in state_dict.keys() if k.startswith("encoder.")]
            )
            feedforward_layers = len(
                [
                    k
                    for k in state_dict.keys()
                    if k.startswith("feedforward.network.") and ".weight" in k
                ]
            )
            decoder_layers = len(
                [k for k in state_dict.keys() if k.startswith("decoder.")]
            )
            global_encoder_layers = len(
                [k for k in state_dict.keys() if k.startswith("global_encoder.")]
            )

            fingerprint = {
                "encoder_layers": encoder_layers,
                "feedforward_layers": feedforward_layers,
                "decoder_layers": decoder_layers,
                "global_encoder_layers": global_encoder_layers,
                "total_params": len(state_dict),
            }

            return fingerprint
        except Exception as e:
            print(f"⚠️ Warning: Could not extract architecture fingerprint: {e}")
            return {}

    def _search_fold_checkpoint(
        self, trial_number: int, fold_idx: int, study_name: str
    ):
        """
        Search for checkpoint file for specific trial and fold.

        Args:
            trial_number: Optuna trial number
            fold_idx: Fold index (0 to n_folds-1)
            study_name: Optional study name to restrict search (recommended to avoid cross-study conflicts)

        Returns:
            str or None: Path to best checkpoint or None if not found
        """
        # Use study name if provided to avoid finding checkpoints from wrong study
        if study_name == None:
            raise ValueError("study_name must be provided to search for checkpoints.")

        study_pattern = f"Study_{study_name}"
        folderPattern = f"{study_name}_trial_{trial_number}_fold_{fold_idx}"

        # Search patterns:
        # 1. New format (no version directory): FinalPipelineLogs/.../checkpoints/
        # 2. Old format (with version_0): FinalPipelineLogs/.../version_0/checkpoints/
        patterns = [
            f"FinalPipelineLogs/{study_pattern}/{folderPattern}/checkpoints/trial-{trial_number}-fold-{fold_idx}-*.ckpt",
            f"FinalPipelineLogs/{study_pattern}/{folderPattern}/version_0/checkpoints/trial-{trial_number}-fold-{fold_idx}-*.ckpt",
        ]

        for pattern in patterns:
            checkpoints = glob.glob(pattern, recursive=True)
            if checkpoints:
                # Select checkpoint with highest val_F1
                if any("val_F1=" in ckpt for ckpt in checkpoints):
                    return sorted(
                        [c for c in checkpoints if "val_F1=" in c],
                        key=lambda x: float(x.split("val_F1=")[1].split(".ckpt")[0]),
                        reverse=True,
                    )[0]
                return checkpoints[0]

        return None

    def _load_fold_model(
        self,
        checkpoint_path,
        archParams,
        macroArch,
        data_loader,
        fold_idx,
        expected_f1,
        best_params,
        threshold=0.95,
    ):
        """
        Load and validate fold model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            expected_f1: Expected F1 score from original training
            threshold: Minimum acceptable ratio (default 0.95)

        Returns:
            tuple: (model, actual_f1) if valid, None if degraded/failed
        """
        try:
            # Load model
            use_windowing = best_params.get("use_windowing", False)
            
            # DEBUG: Check if checkpoint has base_model prefix (indicates WindowedModelWrapper)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict_keys = list(checkpoint["state_dict"].keys())
            has_base_model_prefix = any(key.startswith("base_model.") for key in state_dict_keys)
            
            # Auto-detect windowing from checkpoint structure if parameter is missing
            if has_base_model_prefix and not use_windowing:
                print(f"  [AUTO-DETECT] Checkpoint has 'base_model.' prefix - enabling windowing")
                use_windowing = True

            if use_windowing:
                # Load wrapped model
                window_size = best_params.get("window_size", 160)
                stride_ratio = best_params.get("stride_ratio", 1.0)
                stride = int(window_size * stride_ratio)
                aggregation_method = best_params.get("aggregation_method", "avg_probs")
                window_loss_weight = best_params.get("window_loss_weight", 0.0)

                # Create base model for initialization
                # CRITICAL: Deep copy archParams to prevent mutation
                archParams_copy = copy.deepcopy(archParams)
                base_model = (
                    Direct(params=archParams_copy)
                    if macroArch == "Direct"
                    else LightningAutoencoder(params=archParams_copy)
                )

                model = WindowedModelWrapper.load_from_checkpoint(
                    checkpoint_path,
                    base_model=base_model,
                    window_size=window_size,
                    stride=stride,
                    aggregation_method=aggregation_method,
                    window_loss_weight=window_loss_weight,
                )
            else:
                # Load base model
                # CRITICAL: Deep copy archParams to prevent mutation
                archParams_copy = copy.deepcopy(archParams)
                if macroArch == "Direct":
                    model = Direct.load_from_checkpoint(
                        checkpoint_path, params=archParams_copy
                    )
                else:
                    model = LightningAutoencoder.load_from_checkpoint(
                        checkpoint_path, params=archParams_copy
                    )

            # Validate performance
            print(f"Validating fold {fold_idx} (expected F1: {expected_f1:.4f})...")

            includeTestInTrain = macroArch == "Autoencoder"
            data_loader.setup_fold(fold_idx, include_test_in_train=includeTestInTrain)
            val_loader = data_loader.val_dataloader()

            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Compute F1 with torchmetrics
            f1_metric = F1Score(task="multiclass", num_classes=3, average="macro").to(
                device
            )

            with torch.no_grad():
                for batch in val_loader:
                    (timeSeries, globalFeatures), labels = batch
                    timeSeries = timeSeries.to(device)
                    globalFeatures = globalFeatures.to(device)
                    labels = labels.to(device)

                    outputs = model((timeSeries, globalFeatures))
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    preds = torch.argmax(logits, dim=1)

                    f1_metric.update(preds, labels)

            actual_f1 = f1_metric.compute().item()

            # Check if expected_f1 is valid (should be between 0 and 1)
            # If > 1, it's likely a loss value, not F1 - skip degradation check
            if expected_f1 > 1.0:
                print(
                    f"⚠️ Fold {fold_idx}: Expected F1={expected_f1:.4f} is invalid (>1.0, likely a loss value)"
                )
                print(f"   Skipping degradation check. Actual F1={actual_f1:.4f}")
                return model, actual_f1

            min_acceptable_f1 = expected_f1 * threshold

            if actual_f1 >= min_acceptable_f1:
                print(f"✓ Fold {fold_idx} validated: F1={actual_f1:.4f}")
                return model, actual_f1
            else:
                print(
                    f"⚠️ Fold {fold_idx} degraded: F1={actual_f1:.4f} < {min_acceptable_f1:.4f}"
                )
                return None

        except Exception as e:
            print(f"❌ Error loading fold {fold_idx}:")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Error message: {e}")
            import traceback

            print(f"   Traceback:\n{traceback.format_exc()}")
            return None

    def _retrain_fold_model(
        self, trial_wrapper, archParams, macroArch, data_loader, fold_idx, best_params
    ):
        """
        Retrain model on specific fold.

        Args:
            fold_idx: Fold index to train on

        Returns:
            tuple: (trained_model, val_f1)
        """
        print(f"\n🔄 Retraining fold {fold_idx}...")
        print(f"   MacroArchitecture: {macroArch}")
        print(f"   Use windowing: {best_params.get('use_windowing', True)}")

        try:
            # Create base model
            # CRITICAL: Deep copy archParams to prevent mutation
            archParams_copy = copy.deepcopy(archParams)
            base_model = (
                Direct(params=archParams_copy)
                if macroArch == "Direct"
                else LightningAutoencoder(params=archParams_copy)
            )

            # Apply He initialization
            ff_activation = archParams["FeedForwardParams"]["activation_function"]
            self.apply_he_initialization(base_model, activation_type=ff_activation)

            # Wrap if windowing
            use_windowing = best_params.get("use_windowing", True)
            if use_windowing:
                window_size = best_params.get("window_size", 160)
                stride_ratio = best_params.get("stride_ratio", 1.0)
                stride = int(window_size * stride_ratio)
                aggregation_method = best_params.get("aggregation_method", "avg_probs")
                window_loss_weight = best_params.get("window_loss_weight", 0.0)

                training_model = WindowedModelWrapper(
                    base_model=base_model,
                    window_size=window_size,
                    stride=stride,
                    aggregation_method=aggregation_method,
                    window_loss_weight=window_loss_weight,
                )
            else:
                training_model = base_model

            # Setup fold data
            print(f"   Setting up fold data...")
            includeTestInTrain = macroArch == "Autoencoder"
            data_loader.setup_fold(fold_idx, include_test_in_train=includeTestInTrain)
            train_loader = data_loader.train_dataloader()
            val_loader = data_loader.val_dataloader()
            print(
                f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
            )

            # Training parameters
            max_epochs = 100
            early_stopping_patience = best_params.get("EarlyStoppingPatience", 15)

            # Setup callbacks
            early_stopping_callback = EarlyStopping(
                monitor="val_F1",
                patience=early_stopping_patience,
                mode="max",
                verbose=False,
            )

            checkpoint_callback = ModelCheckpoint(
                monitor="val_F1",
                mode="max",
                save_top_k=1,
                filename=f"retrained-fold-{fold_idx}-" "{epoch:02d}-{val_F1:.3f}",
                verbose=False,
            )

            # Create trainer
            print(f"   Starting training (max {max_epochs} epochs)...")
            trainer = Trainer(
                max_epochs=max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[early_stopping_callback, checkpoint_callback],
                enable_checkpointing=True,
            )

            # Train
            trainer.fit(training_model, train_loader, val_loader)

            # Get best F1 score from training
            best_f1 = (
                checkpoint_callback.best_model_score.item()
                if checkpoint_callback.best_model_score
                else 0.0
            )
            print(f"   Training completed. Best F1: {best_f1:.4f}")

            # Return the trained model directly (no need to reload from checkpoint)
            # The model is already in memory and at its best state
            print(f"✓ Fold {fold_idx} retrained successfully: F1={best_f1:.4f}")
            return training_model, best_f1

        except Exception as e:
            print(f"❌ Error retraining fold {fold_idx}:")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Error message: {e}")
            import traceback

            print(f"   Traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to stop execution

    def load_ensemble_model(self, performance_threshold=0.95, force_retrain=False):
        """
        Load K-fold ensemble from best trial with validation and retraining fallback.

        Args:
            performance_threshold: Minimum acceptable performance ratio (default 0.95)
            force_retrain: If True, skip checkpoint loading and retrain all folds (default False)

        Returns:
            EnsembleModel: Weighted ensemble ready for inference
        """
        print("\n" + "=" * 60)
        print("LOADING K-FOLD ENSEMBLE")
        print("=" * 60)

        # Get best trial
        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            raise ValueError("No completed trials found in study")

        best_trial = self.study.best_trial
        best_params = best_trial.params

        print(f"Best trial: {best_trial.number}")
        print(f"Best mean F1: {best_trial.value:.4f}")
        print(f"Architecture: {best_params.get('MacroArchitecture', 'Unknown')}")
        print("-" * 60)

        # Reconstruct architecture (same as load_best_model)
        class TrialWrapper:
            def __init__(self, params):
                self.params = params

            def suggest_categorical(self, name, choices):
                return self.params[name]

            def suggest_int(self, name, low, high, log=False):
                return self.params[name]

            def suggest_float(self, name, low, high, log=False):
                return self.params[name]

        trial_wrapper = TrialWrapper(best_params)

        includeTestInTrain = best_params.get("MacroArchitecture") == "Autoencoder"
        data_loader = DataModule(params=self.data_params)
        # Load training data and prepare for K-fold splits
        # If includeTestInTrain=True (autoencoder), test data will be loaded WITH augmentation for training
        data_loader.setup(stage="fit", includeTestInTrain=includeTestInTrain)
        datasetInfo = data_loader.getDatasetInfo()
        
        # CRITICAL FIX: When using WindowedModelWrapper, the base model receives windowed sequences
        # So we need to adjust datasetInfo to reflect what the base model actually sees
        use_windowing = best_params.get("use_windowing", False)
        if use_windowing:
            window_size = best_params.get("window_size", 10)
            stride = best_params.get("stride", 5)
            datasetInfo["timeSeriesShape"] = (datasetInfo["timeSeriesShape"][0], window_size)
            datasetInfo["use_windowing"] = True
            datasetInfo["window_size"] = window_size
            datasetInfo["stride"] = stride

        print("Reconstructing architecture...")
        archParams = {}
        archParams = self.setUpEncoder(trial_wrapper, archParams, datasetInfo)
        archParams = self.setUpFeedForwardHead(trial_wrapper, archParams, datasetInfo)

        macroArch = best_params.get("MacroArchitecture", "Direct")
        if macroArch == "Autoencoder":
            archParams = self.setUpDecoder(trial_wrapper, archParams, datasetInfo)
            archParams["ReconstructionLossWeight"] = best_params.get(
                "ReconstructionLossWeight", 0.5
            )

        archParams["LearningRate"] = 0.0005  # FIXED VALUE (must match training)
        archParams["RegularizationWeight"] = best_params["RegularizationWeight"]
        archParams["OutputDim"] = 3
        archParams["ClassWeightsPath"] = self.data_params.get(
            "class_weights_path", "../dataset/PirateProcessed/class_weights.yaml"
        )

        # Add scheduler parameters - using FIXED VALUES from training
        scheduler_type = "ReduceLROnPlateau"  # FIXED VALUE (must match training)
        archParams["SchedulerType"] = scheduler_type
        if scheduler_type == "ReduceLROnPlateau":
            archParams["Patience"] = 10  # FIXED VALUE (must match training)
            archParams["SchedulerFactor"] = 0.1  # FIXED VALUE (must match training)
            archParams["SchedulerMinLR"] = 0  # FIXED VALUE (must match training)
        elif scheduler_type == "CosineAnnealing":
            archParams["T_max"] = 100
            archParams["eta_min"] = best_params.get("eta_min", 1e-6)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            archParams["T_0"] = best_params.get("T_0", 10)
            archParams["T_mult"] = best_params.get("T_mult", 2)
            archParams["eta_min"] = best_params.get("eta_min", 1e-6)

        print("✓ Architecture reconstructed")

        n_folds = self.data_params["n_folds"]
        # Get expected fold scores
        expected_fold_f1s = [best_trial.user_attrs[f"fold_{foldNum}"] for foldNum in range(n_folds) if f"fold_{foldNum}" in best_trial.user_attrs]

        print(f"\nEnsemble details:")
        for foldNum, fold_score in enumerate(expected_fold_f1s):
            print(f"  Fold {foldNum}: Expected F1 Score = {fold_score:.4f}")
        print("-" * 60)

        # ==================== STEP 1: Find all checkpoints ====================
        print(f"\n[STEP 1/3] Searching for checkpoints...")
        print("-" * 60)
        full_study_name = self.project_name + self.study_name
        found_checkpoints = []

        for fold_idx in range(n_folds):
            ckpt_path = self._search_fold_checkpoint(
                best_trial.number, fold_idx, full_study_name
            )
            found_checkpoints.append(ckpt_path)
            if ckpt_path:
                print(f"  Fold {fold_idx}: Found @ {ckpt_path}")
            else:
                print(f"  Fold {fold_idx}: Not found")

        print(
            f"\nCheckpoints found: {sum(1 for cp in found_checkpoints if cp is not None)}/{n_folds}"
        )

        # ==================== STEP 2: Validate architecture consistency ====================
        # Only validate if we have multiple checkpoints and not forcing retrain
        if not force_retrain:
            valid_checkpoints = [cp for cp in found_checkpoints if cp is not None]
            if len(valid_checkpoints) > 1:
                print(f"\n[STEP 2/3] Validating architecture consistency...")
                print("-" * 60)
                fingerprints = [
                    self._get_architecture_fingerprint(cp) for cp in valid_checkpoints
                ]

                # Check if all fingerprints match
                reference_fp = fingerprints[0]
                inconsistent = False
                for i, (cp, fp) in enumerate(
                    zip(valid_checkpoints[1:], fingerprints[1:]), 1
                ):
                    if fp != reference_fp:
                        inconsistent = True
                        print(f"[ERROR] Architecture mismatch detected!")
                        print(f"  Checkpoint 0: {valid_checkpoints[0]}")
                        print(f"    Architecture: {reference_fp}")
                        print(f"  Checkpoint {i}: {cp}")
                        print(f"    Architecture: {fp}")

                if inconsistent:
                    raise ValueError(
                        f"Architecture inconsistency detected across fold checkpoints!\n"
                        f"This suggests checkpoints are from different training runs.\n"
                        f"Solution: Use force_retrain=True to retrain all folds with consistent architecture."
                    )
                else:
                    print(
                        f"[OK] All {len(valid_checkpoints)} checkpoints have consistent architecture"
                    )
                    print(f"  Architecture: {reference_fp}")
            elif len(valid_checkpoints) == 1:
                print(
                    f"\n[STEP 2/3] Skipping consistency check (only 1 checkpoint found)"
                )
                fp = self._get_architecture_fingerprint(valid_checkpoints[0])
                print(f"  Single checkpoint architecture: {fp}")
            else:
                print(f"\n[STEP 2/3] Skipping consistency check (no checkpoints found)")
        else:
            print(f"\n[STEP 2/3] Skipping consistency check (force_retrain=True)")

        # ==================== STEP 3: Load or retrain each fold ====================
        print(f"\n[STEP 3/3] Loading/retraining {n_folds} fold models...")
        print("-" * 60)

        fold_models = []
        fold_f1s = []

        for fold_idx in range(n_folds):
            expected_f1 = (
                expected_fold_f1s[fold_idx]
                if fold_idx < len(expected_fold_f1s)
                else None
            )

            ckpt_path = found_checkpoints[fold_idx]

            # Try to load checkpoint if available and not forcing retrain
            if ckpt_path and expected_f1 is not None and not force_retrain:
                result = self._load_fold_model(
                    ckpt_path,
                    archParams,
                    macroArch,
                    data_loader,
                    fold_idx,
                    expected_f1,
                    best_params,
                    performance_threshold,
                )

                if result is not None:
                    model, f1 = result
                    fold_models.append(model)
                    fold_f1s.append(f1)
                    continue

            # Retrain (checkpoint missing, validation failed, or forced)
            reason = (
                "force_retrain=True"
                if force_retrain
                else ("Checkpoint not found" if not ckpt_path else "Validation failed")
            )
            print(f"Fold {fold_idx}: {reason} - retraining...")
            model, f1 = self._retrain_fold_model(
                trial_wrapper, archParams, macroArch, data_loader, fold_idx, best_params
            )
            fold_models.append(model)
            fold_f1s.append(f1)

        print("-" * 60)
        print(f"[OK] All {n_folds} models loaded/retrained")
        print(f"Fold F1 scores: {[f'{f1:.4f}' for f1 in fold_f1s]}")
        print(f"Average F1 score: {sum(fold_f1s) / len(fold_f1s):.4f}")

        # Create weighted ensemble
        ensemble = EnsembleModel(models=fold_models, weights=fold_f1s)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble = ensemble.to(device)
        ensemble.eval()

        print(f"\n✓ Ensemble created with performance-based weights")
        print(f"Weights: {[f'{w:.3f}' for w in ensemble.weights]}")
        print(f"Weighted F1 Estimate: {sum([f1 * w for f1, w in zip(fold_f1s, ensemble.weights)]):.4f}")
        print(f"Device: {device}")
        print("=" * 60)

        return ensemble

    def load_best_model(self):
        """
        Load the best model from the Optuna study.

        This function reconstructs the model architecture from the best trial's
        hyperparameters and loads the corresponding checkpoint from disk.
        If no checkpoint is found, it falls back to retraining the model.

        Process:
        1. Get best trial from Optuna study
        2. Reconstruct architecture parameters from trial hyperparameters
        3. Search for checkpoint file using multiple strategies:
           - Look for trial number in checkpoint filename
           - Try version directory matching trial number
           - Search nearby version numbers (trials may be sequential)
        4a. If checkpoint found: Load model from checkpoint
        4b. If checkpoint NOT found: Retrain model with best hyperparameters
        5. Apply He initialization
        6. Wrap with WindowedModelWrapper if windowing was used
        7. Set model to evaluation mode

        Returns:
            torch.nn.Module: The loaded best model ready for inference

        Raises:
            ValueError: If no completed trials exist in the study

        Side Effects:
            - Sets self.best_model to the loaded model
            - Prints status messages during loading process
            - May retrain model if checkpoint not found (creates new checkpoint)

        Example:
            >>> pipeline.load_best_model()
            >>> predictions = pipeline.create_submission()
        """
        import glob

        # Get best trial from study
        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            raise ValueError(
                "No completed trials found in study. Run optimization first."
            )

        best_trial = self.study.best_trial
        best_params = best_trial.params

        print("=" * 60)
        print("LOADING BEST MODEL")
        print("=" * 60)
        print(f"Best trial: {best_trial.number}")
        print(f"Best F1 score: {best_trial.value:.4f}")
        print(f"Architecture: {best_params.get('MacroArchitecture', 'Unknown')}")
        print(f"Encoder type: {best_params.get('architectureType', 'Unknown')}")
        print("-" * 60)

        # Create a trial wrapper to reuse setup functions
        class TrialWrapper:
            def __init__(self, params):
                self.params = params

            def suggest_categorical(self, name, choices):
                return self.params[name]

            def suggest_int(self, name, low, high, log=False):
                return self.params[name]

            def suggest_float(self, name, low, high, log=False):
                return self.params[name]

        trial_wrapper = TrialWrapper(best_params)

        # Setup data to get dataset info
        includeTestInTrain = best_params.get("MacroArchitecture") == "Autoencoder"
        data_loader = DataModule(params=self.data_params)
        # First load test data if we need it for autoencoder training
        if includeTestInTrain:
            data_loader.setup(stage="test", includeTestInTrain=False)
        # Then load training data
        data_loader.setup(stage="fit", includeTestInTrain=includeTestInTrain)
        datasetInfo = data_loader.getDatasetInfo()
        
        # CRITICAL FIX: When using WindowedModelWrapper, the base model receives windowed sequences
        # So we need to adjust datasetInfo to reflect what the base model actually sees
        use_windowing = best_params.get("use_windowing", False)
        if use_windowing:
            window_size = best_params.get("window_size", 10)
            stride = best_params.get("stride", 5)
            datasetInfo["timeSeriesShape"] = (datasetInfo["timeSeriesShape"][0], window_size)
            datasetInfo["use_windowing"] = True
            datasetInfo["window_size"] = window_size
            datasetInfo["stride"] = stride

        # Reconstruct architecture parameters
        print("Reconstructing architecture...")
        archParams = {}
        archParams = self.setUpEncoder(trial_wrapper, archParams, datasetInfo)
        archParams = self.setUpFeedForwardHead(trial_wrapper, archParams, datasetInfo)

        macroArch = best_params.get("MacroArchitecture", "Direct")
        if macroArch == "Autoencoder":
            archParams = self.setUpDecoder(trial_wrapper, archParams, datasetInfo)
            archParams["ReconstructionLossWeight"] = best_params.get(
                "ReconstructionLossWeight", 0.5
            )

        # Add training parameters - using FIXED VALUES from training
        archParams["LearningRate"] = 0.0005  # FIXED VALUE (must match training)
        archParams["RegularizationWeight"] = best_params["RegularizationWeight"]
        archParams["OutputDim"] = 3
        useClassWeights = best_params.get("useClassWeights", True)
        archParams["ClassWeightsPath"] = self.data_params.get(
            "class_weights_path", "../dataset/PirateProcessed/class_weights.yaml" if useClassWeights else None
        )

        # Add scheduler parameters - using FIXED VALUES from training
        scheduler_type = "ReduceLROnPlateau"  # FIXED VALUE (must match training)
        archParams["SchedulerType"] = scheduler_type

        if scheduler_type == "ReduceLROnPlateau":
            archParams["Patience"] = 10  # FIXED VALUE (must match training)
            archParams["SchedulerFactor"] = 0.1  # FIXED VALUE (must match training)
            archParams["SchedulerMinLR"] = 0  # FIXED VALUE (must match training)
        elif scheduler_type == "CosineAnnealing":
            archParams["T_max"] = 100
            archParams["eta_min"] = best_params.get("eta_min", 1e-6)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            archParams["T_0"] = best_params.get("T_0", 10)
            archParams["T_mult"] = best_params.get("T_mult", 2)
            archParams["eta_min"] = best_params.get("eta_min", 1e-6)

        print("✓ Architecture reconstructed")

        # Find checkpoint file
        print("Searching for checkpoint...")
        checkpoint_path = None

        # Strategy 1: Look for trial number in checkpoint filename (in new logs location)
        checkpoint_pattern = (
            f"FinalPipelineLogs/Study_*/*/checkpoints/trial-{best_trial.number}-*.ckpt"
        )
        checkpoints = glob.glob(checkpoint_pattern)

        # Strategy 2: Search all checkpoints with trial number (any study)
        if not checkpoints:
            checkpoint_pattern = (
                f"FinalPipelineLogs/**/trial-{best_trial.number}-*.ckpt"
            )
            checkpoints = glob.glob(checkpoint_pattern, recursive=True)

        # Strategy 3: Fallback to old lightning_logs location
        if not checkpoints:
            checkpoint_pattern = (
                f"lightning_logs/version_*/checkpoints/trial-{best_trial.number}-*.ckpt"
            )
            checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            # FALLBACK: Retrain the model if no checkpoint found
            print(
                "⚠️ No checkpoint found. Retraining best model with optimal hyperparameters..."
            )
            print("-" * 60)

            base_model = self._retrain_best_model(
                trial_wrapper, archParams, macroArch, data_loader, best_params
            )

        else:
            # Select best checkpoint (prefer ones with val_F1 in name)
            if any("val_F1=" in ckpt for ckpt in checkpoints):
                checkpoint_path = sorted(
                    [c for c in checkpoints if "val_F1=" in c],
                    key=lambda x: float(x.split("val_F1=")[1].split(".ckpt")[0]),
                    reverse=True,
                )[0]
            else:
                checkpoint_path = sorted(checkpoints)[-1]

            print(f"✓ Found checkpoint: {checkpoint_path}")

            # Check if windowing was used
            use_windowing = True

            # Try to load model from checkpoint
            print("Loading model from checkpoint...")
            try:
                # Load checkpoint to inspect its structure
                checkpoint = torch.load(checkpoint_path, weights_only=False)

                # Check if checkpoint has hyper_parameters (saved by Lightning)
                if "hyper_parameters" in checkpoint:
                    print("✓ Found saved hyperparameters in checkpoint")
                    saved_params = checkpoint["hyper_parameters"].get(
                        "params", archParams
                    )

                    # Use saved params if available, otherwise use reconstructed ones
                    if saved_params != archParams:
                        print(
                            "⚠️ Using parameters from checkpoint (differ from reconstructed)"
                        )
                        archParams = saved_params

                if use_windowing:
                    # If windowing was used, we need to load into WindowedModelWrapper
                    print(
                        "Checkpoint was saved with WindowedModelWrapper, loading accordingly..."
                    )

                    # First create base model with checkpoint params
                    # CRITICAL: Deep copy archParams to prevent mutation
                    archParams_copy = copy.deepcopy(archParams)
                    if macroArch == "Direct":
                        base_model = Direct(params=archParams_copy)
                    else:
                        base_model = LightningAutoencoder(params=archParams_copy)

                    # Apply He initialization
                    ff_activation = archParams["FeedForwardParams"][
                        "activation_function"
                    ]
                    self.apply_he_initialization(
                        base_model, activation_type=ff_activation
                    )

                    # Create WindowedModelWrapper
                    window_size = best_params.get("window_size", 160)
                    stride_ratio = best_params.get("stride_ratio", 1.0)
                    stride = int(window_size * stride_ratio)
                    aggregation_method = best_params.get(
                        "aggregation_method", "avg_probs"
                    )
                    window_loss_weight = best_params.get("window_loss_weight", 0.0)

                    wrapped_model = WindowedModelWrapper(
                        base_model=base_model,
                        window_size=window_size,
                        stride=stride,
                        aggregation_method=aggregation_method,
                        window_loss_weight=window_loss_weight,
                    )

                    # Load checkpoint into wrapped model
                    missing_keys, unexpected_keys = wrapped_model.load_state_dict(
                        checkpoint["state_dict"], strict=False
                    )

                    if missing_keys:
                        print(
                            f"⚠️ Warning: Missing keys in checkpoint: {len(missing_keys)} keys"
                        )
                        if len(missing_keys) <= 10:
                            for key in missing_keys:
                                print(f"  - {key}")
                    if unexpected_keys:
                        print(
                            f"⚠️ Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys"
                        )
                        if len(unexpected_keys) <= 10:
                            for key in unexpected_keys:
                                print(f"  - {key}")

                    model = wrapped_model

                else:
                    # No windowing, load directly into base model
                    if macroArch == "Direct":
                        base_model = Direct.load_from_checkpoint(
                            checkpoint_path, params=archParams
                        )
                    else:
                        base_model = LightningAutoencoder.load_from_checkpoint(
                            checkpoint_path, params=archParams
                        )

                    # Apply He initialization
                    ff_activation = archParams["FeedForwardParams"][
                        "activation_function"
                    ]
                    self.apply_he_initialization(
                        base_model, activation_type=ff_activation
                    )

                    model = base_model

                print("✓ Model loaded successfully from checkpoint")

            except Exception as e:
                # FALLBACK: If checkpoint loading fails, retrain the model
                print(f"⚠️ Error loading checkpoint: {e}")
                print("Retraining best model with optimal hyperparameters...")
                print("-" * 60)

                base_model = self._retrain_best_model(
                    trial_wrapper, archParams, macroArch, data_loader, best_params
                )

                # Wrap with WindowedModelWrapper if windowing was used
                if use_windowing:
                    print("Wrapping retrained model with WindowedModelWrapper...")
                    window_size = best_params.get("window_size", 160)
                    stride_ratio = best_params.get("stride_ratio", 1.0)
                    stride = int(window_size * stride_ratio)
                    aggregation_method = best_params.get(
                        "aggregation_method", "avg_probs"
                    )
                    window_loss_weight = best_params.get("window_loss_weight", 0.0)

                    model = WindowedModelWrapper(
                        base_model=base_model,
                        window_size=window_size,
                        stride=stride,
                        aggregation_method=aggregation_method,
                        window_loss_weight=window_loss_weight,
                    )
                else:
                    model = base_model

        # Set to eval mode and move to device
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"✓ Model loaded and moved to {device}")
        print("=" * 60)

        # Store model and dataloader
        self.best_model = model
        self.dataloader = data_loader

        return model

    # submissions
    def create_submission(self):
        """
        Create submission file for test dataset predictions.

        This function loads the best trained model, performs inference on the
        test dataset, and generates a submission CSV file in the required format.

        Steps:
        1. Load the best model checkpoint from training (if not already loaded).
        2. Prepare the test dataset using the same preprocessing as training.
        3. Perform inference to get predicted class labels.
        4. Format predictions into a DataFrame with required columns.
        5. Save the DataFrame as 'submission.csv'.

        Note:
            - Ensure that the model architecture and preprocessing match those used during training.
            - The submission file should contain columns: 'sample_index', 'label'.
            - If best_model is not loaded, will automatically call load_best_model()

        Returns:
            pd.DataFrame: The submission dataframe
        """
        # Load K-fold ensemble if not already loaded
        if self.best_model is None:
            print("Loading K-fold ensemble...")
            self.best_model = self.load_ensemble_model()
            self.dataloader = DataModule(params=self.data_params)

        if self.dataloader is None:
            raise ValueError("No dataloader found.")

        # Setup test dataloader
        self.dataloader.setup(stage="test", includeTestInTrain=False)
        test_loader = self.dataloader.test_dataloader()

        # Create submission generator
        submitter = WindowedSubmissionGenerator(
            model=self.best_model,
            dataloader=test_loader,
            label_mapping={0: "no_pain", 1: "low_pain", 2: "high_pain"},
            aggregation_method="avg_probs",
        )

        # Generate timestamp and output path
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.submission_folder, f"submission_{time_now}.csv")

        # Generate submission
        submission_df = submitter.generate_submission(output_path=path)

        print(f"✓ Submission generated: {path}")
        print(f"Total predictions: {len(submission_df)}")

        return submission_df
