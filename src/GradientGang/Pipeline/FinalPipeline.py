import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import F1Score
import warnings
import dotenv
import os
import numpy as np

warnings.filterwarnings('ignore')

from GradientGang.Pipeline.DataLoader.DataLoader import DataModule
from GradientGang.Pipeline.Architectures.Direct import Direct
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder
from GradientGang.Pipeline.Utils.ParameterInterpreter import ParameterInterpreter


class FinalPipeline:
    """
    End-to-end pipeline for neural network hyperparameter optimization using Optuna.
    """

    def __init__(self, params: dict):
        """
        Initialize the FinalPipeline with configuration parameters.
        
        Args:
            params (dict): Configuration dictionary containing:
                - project_name (str): Name for the optimization project
                - data_params (dict): Data loading parameters including:
                    - data_dir (str): Directory containing data files
                    - train_file_name (str): Training data filename
                    - train_file_name_labels (str): Training labels filename
                    - test_file_name (str): Test data filename
                    - train_global_features_file (str): Global features for training
                    - test_global_features_file (str): Global features for testing
                    - batch_size (int): Batch size for training
                    - num_workers (int): Number of data loading workers
                    - val_split (float): Validation split ratio
                    - shuffle (bool): Whether to shuffle data
                    - use_kfold (bool): Whether to use K-fold cross-validation
                    - n_folds (int): Number of folds for cross-validation
                - database_path (str): Path to .env file with database URL
                - study_name (str): Name for the Optuna study
        
        Raises:
            ValueError: If required parameters (project_name, data_params, 
                       database_path, study_name) are missing
        """
        self.params = params
        self.storage = None
        self.dataloader = None
        self.study = None

        # Initialize project namee
        self.project_name = params.get("project_name", None)
        if not self.project_name:
            raise ValueError("Project name must be provided in parameters.")

        # Initialize data parameters
        self.data_params = params.get("data_params", None)
        if not self.data_params:
            raise ValueError("Data parameters must be provided in parameters.")

        # Initialize database connection
        self.database_path = params.get("database_path", None)
        if not self.database_path:
            raise ValueError("Database path must be provided in parameters.")
        self.load_database()
        print("✓ Database initialized successfully!")

        # Initialize study
        self.study_name = params.get("study_name", None)
        if not self.study_name:
            raise ValueError("Study name must be provided in parameters.")
        self.create_study()
        print("✓ Study initialized successfully!")
        print("✓ FinalPipeline initialized successfully!")

    #DB management
    def load_database(self):
        """
        Load database configuration from environment file.
        
        Reads the DATABASE_URL from the .env file specified in database_path.
        This URL is used for persistent storage of Optuna study results.
        """
        dotenv.load_dotenv(dotenv_path=self.database_path)
        self.storage = os.getenv("DATABASE_URL")
        print(f"Storage: {self.storage[:21]}..." if self.storage else "⚠️ No database URL found")
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
            direction='maximize',  # Maximize F1 score
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,  # Number of trials before pruning starts
                n_warmup_steps=5,     # Steps to wait before checking for pruning
                interval_steps=1      # Check pruning at every step
            ),
            study_name=self.project_name + self.study_name,
            storage=self.storage,
            load_if_exists=True  # Resume from existing study if available
        )

        print("✓ Study created/loaded successfully!")
        print(f"Study name: {self.study.study_name}")
        print(f"Sampler: {self.study.sampler.__class__.__name__}")
        print(f"Pruner: {self.study.pruner.__class__.__name__}")
        print(f"Storage: {'Database' if self.storage else 'In-memory'}")
        print(f"Total trials: {len(self.study.trials)}")

    #optimizer
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
            self.objective_kfold, 
            show_progress_bar=True, 
            n_trials=n_trials
        )

        print("\nOptimization completed!")
        print(f"Best trial: {self.study.best_trial.number}")
        print(f"Best F1 score: {self.study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

    # Architecture Setup Methods
    def setUpEncoder(self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict):
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
        globalEmbeddingDim = trial.suggest_int("globalEmbeddingDim", 16, 128)
        globalNumLayers = trial.suggest_int("globalNumLayers", 1, 3)
        globalDropout = trial.suggest_float("globalDropout", 0.0, 0.5)
        globalActivation = trial.suggest_categorical("globalActivation", ["ReLU", "LeakyReLU", "GELU"])
        
        # Build global feature encoder layers
        globalLayerList = []
        currentDim = globalInputDim
        for i in range(globalNumLayers):
            nextDim = globalEmbeddingDim if i == globalNumLayers - 1 else trial.suggest_int(f"globalHiddenDim_{i}", 32, 128)
            globalLayerList.append({
                "name": "Linear",
                "params": {
                    "in_features": currentDim,
                    "out_features": nextDim,
                    "bias": True,
                }
            })
            if globalDropout > 0 and i < globalNumLayers - 1:
                globalLayerList.append({
                    "name": "Dropout",
                    "params": {
                        "p": globalDropout,
                        "inplace": False,
                    }
                })
            currentDim = nextDim
        
        globalEncoderParams = {
            "activation_function": globalActivation,
            "layer_type": globalLayerList
        }
        architectureParameters["GlobalFFEncoderParams"] = globalEncoderParams
        
        # Setup time series encoder - now supports Recurrent, Conv1d, and MultiScaleCNN
        architectureType = trial.suggest_categorical("architectureType", ["Recurrent", "Conv1d", "MultiScaleCNN"])
        
        timeSeriesEncoderParams = {}
        
        if architectureType == "Recurrent":
            rnnType = trial.suggest_categorical("rnnType", ["LSTM", "GRU"])
            hiddenDim = trial.suggest_int("hiddenDim", 16, 256)
            numLayers = trial.suggest_int("numLayers", 1, 3)
            bidirectional = trial.suggest_categorical("bidirectional", [False, True])
            dropout = trial.suggest_float("recurrentDropout", 0.0, 0.5)
            activationFunction = trial.suggest_categorical("encoderActivation", ["ReLU", "LeakyReLU", "GELU"])
            
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
                        }
                    },
                ]
            }
            
        elif architectureType == "Conv1d":
            # Conv1d architecture
            numConvLayers = trial.suggest_int("numConvLayers", 1, 3)
            kernelSize = trial.suggest_categorical("kernelSize", [3, 5, 7])
            stride = trial.suggest_categorical("strideConv1D", [1, 2])
            activationFunction = trial.suggest_categorical("encoderActivation", ["ReLU", "LeakyReLU", "GELU"])
            
            # Start with input channels
            inputChannels = datasetInfo["timeSeriesShape"][0]
            
            # Build conv layers
            layerList = []
            currentChannels = inputChannels
            for i in range(numConvLayers):
                outChannels = trial.suggest_int(f"convChannels_{i}", 32, 256)
                layerList.append({
                    "name": "Conv1d",
                    "params": {
                        "in_channels": currentChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSize,
                        "stride": stride,
                        "padding": kernelSize // 2,
                        "bias": True,
                    }
                })
                # Add pooling after each conv
                poolType = trial.suggest_categorical(f"poolType_{i}", ["max", "avg"])
                if poolType == "max":
                    layerList.append({
                        "name": "MaxPool1d",
                        "params": {
                            "kernel_size": 2,
                            "stride": 2,
                        }
                    })
                else:
                    layerList.append({
                        "name": "AvgPool1d",
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
                "activation_function": activationFunction,
                "layer_type": layerList
            }
        
        elif architectureType == "MultiScaleCNN":
            # Multi-scale CNN architecture (Inception-style)
            numMultiScaleLayers = trial.suggest_int("numMultiScaleLayers", 1, 2)
            useDilation = trial.suggest_categorical("useDilation", [True, False])
            kernelSizesProfiles = [(3, 5, 7), (3, 7, 11), (5, 9, 13)]
            kernelSizesIndex = trial.suggest_int("kernelSizesIndex", 0, len(kernelSizesProfiles) - 1)
            kernelSizes = kernelSizesProfiles[kernelSizesIndex]
            activationFunction = trial.suggest_categorical("encoderActivation", ["ReLU", "LeakyReLU", "GELU"])
            
            # Start with input channels
            inputChannels = datasetInfo["timeSeriesShape"][0]
            
            # Build multi-scale layers
            layerList = []
            currentChannels = inputChannels
            for i in range(numMultiScaleLayers):
                branchChannels = trial.suggest_int(f"branchChannels_{i}", 32, 128)
                poolingType = trial.suggest_categorical(f"poolingType_{i}", ["max", "avg", "none"])
                
                layerList.append({
                    "name": "MultiScaleCNN",
                    "params": {
                        "in_channels": currentChannels,
                        "branch_channels": branchChannels,
                        "kernel_sizes": kernelSizes,
                        "use_dilation": useDilation,
                        "pooling_type": poolingType,
                    }
                })
                
                # Update current channels: output is branch_channels * num_branches
                currentChannels = branchChannels * len(kernelSizes)
            
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
                "activation_function": activationFunction,
                "layer_type": layerList
            }
        
        architectureParameters["EncoderParams"] = timeSeriesEncoderParams
        return architectureParameters

    def setUpFeedForwardHead(self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict):
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
        globalEmbeddingDim = architectureParameters["GlobalFFEncoderParams"]["layer_type"][-1]["params"]["out_features"]
        
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
        numHiddenLayers = trial.suggest_int("numFFLayers", 1, 3)
        ffHiddenDim = trial.suggest_int("ffHiddenDim", 32, 256)
        ffDropout = trial.suggest_float("ffDropout", 0.0, 0.5)
        ffActivation = trial.suggest_categorical("ffActivation", ["ReLU", "LeakyReLU", "GELU"])
        
        # Build layer list - make sure last element is always Linear
        layerList_clean = []
        currentDim = combinedInputSize
        for i in range(numHiddenLayers):
            layerList_clean.append({
                "name": "Linear",
                "params": {
                    "in_features": currentDim,
                    "out_features": ffHiddenDim,
                    "bias": True,
                }
            })
            # Add dropout BEFORE the next layer (not after the last one)
            if ffDropout > 0 and i < numHiddenLayers - 1:
                layerList_clean.append({
                    "name": "Dropout",
                    "params": {
                        "p": ffDropout,
                        "inplace": False,
                    }
                })
            currentDim = ffHiddenDim
        
        # Note: Final output layer will be added by Direct/Autoencoder class
        # The last layer MUST have "out_features" for Direct to append the output layer
        feedForwardParams = {
            "activation_function": ffActivation,
            "layer_type": layerList_clean
        }
        
        architectureParameters["FeedForwardParams"] = feedForwardParams
        return architectureParameters

    def setUpDecoder(self, trial: optuna.Trial, architectureParameters: dict, datasetInfo: dict):
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
            
            # RNN layer: input_size must match output features for teacher forcing
            layerList.append({
                "name": rnnType,
                "params": {
                    "input_size": outputSize,  # 34 (for teacher forcing)
                    "hidden_size": hiddenDim,  # Keep encoder's hidden size
                    "num_layers": numLayers,
                    "bias": True,
                    "batch_first": True,
                    "dropout": dropout if numLayers > 1 else 0.0,
                    "bidirectional": False,  # Decoder typically not bidirectional
                }
            })
            
            # Linear projection: hidden_size -> output_size (34 channels)
            layerList.append({
                "name": "Linear",
                "params": {
                    "in_features": hiddenDim,
                    "out_features": outputSize,  # 34
                    "bias": True,
                }
            })
            
            # Now output is (batch, seq_len, 34) which becomes (batch, 34, 160) after permute
            
            decoderParams = {
                "activation_function": activationFunction,
                "layer_type": layerList
            }
            
        elif firstLayer["name"] == "Conv1d":
            # Conv1d decoder - symmetric architecture using ConvTranspose1d
            # The encoder ends with: Conv layers -> AdaptiveAvgPool1d(1) -> Flatten
            # Decoder: Unflatten -> ConvTranspose layers (reversed)
            
            # Get conv layers from encoder
            convLayers = [l for l in encoderParams["layer_type"] if l["name"] == "Conv1d"]
            poolLayers = [l for l in encoderParams["layer_type"] if l["name"] in ["MaxPool1d", "AvgPool1d"]]
            
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
            layerList.append({
                "name": "Unflatten",
                "params": {
                    "dim": 1,
                    "unflattened_size": (lastConvChannels, startSeqLen)
                }
            })
            
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
                    outChannels = reversedConvLayers[i+1]["params"]["out_channels"]
                
                # Add upsampling with ConvTranspose1d
                # Use stride=2 to upsample if there was pooling in encoder
                useStride = 2 if i < numPoolLayers else stride
                
                layerList.append({
                    "name": "ConvTranspose1d",
                    "params": {
                        "in_channels": currentChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSize,
                        "stride": useStride,
                        "padding": kernelSize // 2,
                        "output_padding": useStride - 1 if useStride > 1 else 0,
                        "bias": True,
                    }
                })
                
                # Update current channels for next layer
                currentChannels = outChannels
            
            # Add final adjustment layer to match exact sequence length
            # Use adaptive interpolation if needed
            targetSeqLen = datasetInfo["timeSeriesShape"][1]  # 160
            layerList.append({
                "name": "AdaptiveAvgPool1d",
                "params": {
                    "output_size": targetSeqLen,
                }
            })
            
            decoderParams = {
                "activation_function": encoderParams["activation_function"],
                "layer_type": layerList
            }
        
        elif firstLayer["name"] == "MultiScaleCNN":
            # MultiScaleCNN decoder - symmetric architecture using ConvTranspose1d
            # The encoder ends with: MultiScaleCNN layers -> AdaptiveAvgPool1d(1) -> Flatten
            # Decoder: Unflatten -> ConvTranspose layers (reversed)
            
            # Get MultiScaleCNN layers from encoder
            multiScaleLayers = [l for l in encoderParams["layer_type"] if l["name"] == "MultiScaleCNN"]
            poolLayers = [l for l in encoderParams["layer_type"] if l["name"] in ["MaxPool1d", "AvgPool1d"]]
            
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
            layerList.append({
                "name": "Unflatten",
                "params": {
                    "dim": 1,
                    "unflattened_size": (lastMultiScaleChannels, startSeqLen)
                }
            })
            
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
                    nextLayer = reversedLayers[i+1]
                    outChannels = nextLayer["params"]["branch_channels"] * len(nextLayer["params"]["kernel_sizes"])
                
                # Use ConvTranspose1d for upsampling
                # Use stride=2 to upsample if there was pooling in encoder
                useStride = 2 if i < numPoolLayers else 1
                
                # Use a moderate kernel size for transposed convolution
                kernelSize = 5  # Good balance for upsampling
                
                layerList.append({
                    "name": "ConvTranspose1d",
                    "params": {
                        "in_channels": currentChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSize,
                        "stride": useStride,
                        "padding": kernelSize // 2,
                        "output_padding": useStride - 1 if useStride > 1 else 0,
                        "bias": True,
                    }
                })
                
                # Update current channels for next layer
                currentChannels = outChannels
            
            # Add final adjustment layer to match exact sequence length
            targetSeqLen = datasetInfo["timeSeriesShape"][1]  # 160
            layerList.append({
                "name": "AdaptiveAvgPool1d",
                "params": {
                    "output_size": targetSeqLen,
                }
            })
            
            decoderParams = {
                "activation_function": encoderParams["activation_function"],
                "layer_type": layerList
            }
        
        else:
            # Fallback
            decoderParams = encoderParams
        
        # Mirror global decoder - reverse the encoder architecture
        globalEncoderParams = architectureParameters["GlobalFFEncoderParams"]
        globalEncoderLayers = globalEncoderParams["layer_type"]
        
        # Get only Linear layers from encoder (skip Dropout)
        globalEncoderLinearLayers = [l for l in globalEncoderLayers if l["name"] == "Linear"]
        
        # Reverse the architecture
        globalDecoderLayerList = []
        for i, encoderLayer in enumerate(reversed(globalEncoderLinearLayers)):
            # Swap in_features and out_features
            inFeatures = encoderLayer["params"]["out_features"]
            outFeatures = encoderLayer["params"]["in_features"]
            
            globalDecoderLayerList.append({
                "name": "Linear",
                "params": {
                    "in_features": inFeatures,
                    "out_features": outFeatures,
                    "bias": True,
                }
            })
            
            # Add dropout between layers (not after last)
            if i < len(globalEncoderLinearLayers) - 1:
                # Find dropout from encoder if it exists
                for encLayer in globalEncoderLayers:
                    if encLayer["name"] == "Dropout":
                        globalDecoderLayerList.append({
                            "name": "Dropout",
                            "params": encLayer["params"]
                        })
                        break
        
        globalDecoderParams = {
            "activation_function": globalEncoderParams["activation_function"],
            "layer_type": globalDecoderLayerList
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
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                elif activation_type in ["ReLU", "GELU"]:
                    # For ReLU and GELU
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                # Initialize bias to small constant
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
                    
            elif isinstance(module, torch.nn.Conv1d):
                # He initialization for Conv1d layers
                if activation_type == "LeakyReLU":
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                elif activation_type in ["ReLU", "GELU"]:
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                # Initialize bias to small constant
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
            
            elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
                # For RNN layers, initialize with orthogonal initialization (common practice)
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        # Input-hidden weights: use He initialization
                        torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in param_name:
                        # Hidden-hidden weights: use orthogonal initialization for stability
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in param_name:
                        # Initialize biases to small constant
                        torch.nn.init.constant_(param.data, 0.01)
                        # For LSTM, set forget gate bias to 1 (helps with gradient flow)
                        if isinstance(module, torch.nn.LSTM):
                            n = param.data.size(0)
                            param.data[n//4:n//2].fill_(1.0)  # Forget gate bias

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
        macroArchitecture = trial.suggest_categorical("MacroArchitecture", ["Direct", "Autoencoder"])
        
        # ==================== STEP 2: Configure Data Windowing ====================
        # Windowing splits time series into smaller overlapping segments
        # This can help the model learn from more samples and capture local patterns
        use_windowing = trial.suggest_categorical("use_windowing", [True, False])
        
        if use_windowing:
            window_size = trial.suggest_int("window_size", 10, 160)

            stride = trial.suggest_int("windowStride", 0, window_size)
            # Ensure stride doesn't exceed window_size
            if stride > window_size:
                stride = window_size
        else:
            window_size = 160
            stride = 160
        
        # ==================== STEP 3: Setup Data Loading ====================
        # For autoencoders, include test data in training (unsupervised reconstruction)
        # For direct classification, only use labeled training data
        includeTestInTrain = macroArchitecture == "Autoencoder"
        
        # Create K-Fold data loader with windowing parameters
        kfold_data_params = self.data_params.copy()
        kfold_data_params['use_windowing'] = use_windowing
        kfold_data_params['window_size'] = window_size
        kfold_data_params['stride'] = stride
        n_folds = kfold_data_params['n_folds']
        
        kfold_dataLoader = DataModule(params=kfold_data_params)
        kfold_dataLoader.setup(stage='fit', includeTestInTrain=includeTestInTrain)
        datasetInfo = kfold_dataLoader.getDatasetInfo()
        
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
            archParams["ReconstructionLossWeight"] = trial.suggest_float("ReconstructionLossWeight", 0.1, 0.9)
        
        # ==================== STEP 5: Configure Training Parameters ====================
        # Common parameters for all architectures
        archParams["OutputDim"] = 3  # Number of classes (pain levels)
        archParams["LearningRate"] = trial.suggest_float("LearningRate", 1e-5, 1e-2, log=True)
        archParams["RegularizationWeight"] = trial.suggest_float("RegularizationWeight", 1e-3, 1e1, log=True)
        archParams["ClassWeightsPath"] = "../dataset/PirateProcessed/class_weights.yaml"
        
        # Maximum training epochs (early stopping may terminate earlier)
        max_epochs = 100
        
        # ==================== STEP 6: Configure Learning Rate Scheduler ====================
        # Different schedulers for adaptive learning rate adjustment
        scheduler_type = trial.suggest_categorical("SchedulerType", ["ReduceLROnPlateau", "CosineAnnealing", "CosineAnnealingWarmRestarts"])
        archParams["SchedulerType"] = scheduler_type
        
        if scheduler_type == "ReduceLROnPlateau":
            # Reduce LR when validation metric plateaus
            archParams["Patience"] = trial.suggest_int("SchedulerPatience", 3, 10)
            archParams["SchedulerFactor"] = trial.suggest_float("SchedulerFactor", 0.1, 0.5)
            archParams["SchedulerMinLR"] = trial.suggest_float("SchedulerMinLR", 1e-6, 1e-4, log=True)
        elif scheduler_type == "CosineAnnealing":
            # Cosine annealing: smooth LR decay following cosine curve
            archParams["T_max"] = max_epochs  # Full cycle matches training duration
            archParams["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            # Cosine annealing with periodic restarts (helps escape local minima)
            archParams["T_0"] = trial.suggest_int("T_0", 5, 20)  # Initial restart period
            archParams["T_mult"] = trial.suggest_int("T_mult", 1, 3)  # Period multiplier after restart
            archParams["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
        
        # Early stopping: stops training if no improvement after patience epochs
        early_stopping_patience = trial.suggest_int("EarlyStoppingPatience", 10, 20)
        
        # ==================== STEP 7: K-Fold Cross-Validation Training ====================
        # Track validation scores across all folds
        fold_scores = []
        
        # Train a separate model on each fold to get robust performance estimate
        for fold_idx in range(n_folds):
            # Prepare data for this specific fold (different train/val split)
            kfold_dataLoader.setup_fold(fold_idx, include_test_in_train=includeTestInTrain)
            trainLoader = kfold_dataLoader.train_dataloader()
            valLoader = kfold_dataLoader.val_dataloader()
            
            # Create a fresh model instance for this fold (no weight sharing between folds)r this fold (no weight sharing between folds)
            if macroArchitecture == "Direct":
                model = Direct(archParams)
            else:
                model = LightningAutoencoder(archParams)
            
            # Apply He (Kaiming) initialization for better gradient flow
            ff_activation = archParams["FeedForwardParams"]["activation_function"]
            self.apply_he_initialization(model, activation_type=ff_activation)
            
            # Configure callbacks for training
            # Early stopping: stops training if validation F1 doesn't improve
            early_stopping_callback = EarlyStopping(
                monitor='val_F1',
                patience=early_stopping_patience,
                mode='max',
                verbose=False
            )
            
            # Model checkpointing: saves best model based on validation F1
            checkpoint_callback = ModelCheckpoint(
                monitor='val_F1',
                mode='max',
                save_top_k=1,
                filename=f'trial-{trial.number}-fold-{fold_idx}-' + '{epoch:02d}-{val_F1:.3f}',
                verbose=False
            )
            
            # Create PyTorch Lightning trainer with configured callbacks
            trainer = Trainer(
                max_epochs=max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=20,
                callbacks=[early_stopping_callback, checkpoint_callback],
                enable_checkpointing=True,
            )
            
            # Train the model
            try:
                trainer.fit(model, trainLoader, valLoader)
                
                # Get best validation F1 for this fold
                best_f1 = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score is not None else 0.0
                fold_scores.append(best_f1)
                
            except Exception as e:
                print(f"Fold {fold_idx} failed with error: {e}")
                #Prune the trial if any fold fails
                raise optuna.TrialPruned() from e
        
        # ==================== STEP 8: Calculate Cross-Validation Metrics ====================
        # Aggregate performance across all folds
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        
        # Store detailed results for later analysis
        trial.set_user_attr("fold_scores", fold_scores)  # Individual fold scores
        trial.set_user_attr("mean_f1", mean_f1)          # Mean across folds
        trial.set_user_attr("std_f1", std_f1)            # Standard deviation (stability measure)
        
        # Report score to Optuna for pruning decisions
        trial.report(mean_f1, step=0)
        
        # Check if this trial should be pruned (stopped early)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Return mean F1 as the optimization objective
        return mean_f1
