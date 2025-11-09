"""
hyper.yaml

"param1":
    "type": "categ" | "float" | "int"
    ...
"""

import lightning as L
from .Optimizer.Optimizer import Optimizer
from .Architectures.LightningAutoencoder import LightningAutoencoder
from .Architectures.Direct import Direct
from .Utils.ParameterInterpreter import ParameterInterpreter
import yaml

class Pipeline:
    def __init__(
        self, 
        dataset: L.LightningDataModule, 
        test: L.LightningDataModule, 
        optimizer: Optimizer, 
        dict_config: dict = None, 
        path_config: str = None
    ):
        
        """
        Initializes the Pipeline with dataset, test data, optimizer, and configuration.
        Args:
            dataset (L.LightningDataModule): The training/validation dataset.
            test (L.LightningDataModule): The test dataset.
            optimizer (Optimizer): The optimizer to use for training.
            dict_config (dict, optional): Configuration dictionary. Defaults to None.
            path_config (str, optional): Path to configuration file. Defaults to None.
        Raises:
            ValueError: If both or neither of dict_config and path_config are provided.
        """

        # Validate inputs
        if dict_config == path_config == None or (dict_config != None and path_config != None):
            raise ValueError("dict_config or path_config have to be assigned")
        elif not(isinstance(dataset, L.LightningDataModule) and isinstance(test, L.LightningDataModule)):
            raise ValueError("dataset and test must be L.LightningDataModule")
        elif dict_config is not None and not isinstance(dict_config, dict):
            raise ValueError("dict_config must be a dictionary")
        elif path_config is not None and not isinstance(path_config, str):
            raise ValueError("path_config must be a string")

        # Assign attributes
        self.dataset: L.LightningDataModule = dataset
        self.test: L.LightningDataModule = test
        self.optimizer: Optimizer = optimizer

        # Load configuration
        if dict_config is not None:
            self.config = dict_config
        elif path_config is not None:
            with open(path_config, "r") as f:
                self.config = yaml.safe_load(f)

    def build_architecture(
        self, params: dict
    ) -> L.LightningModule:
        
        # Define architecture interpretation mapping
        interpretation = {
            "autoencoder_joint": LightningAutoencoder, 
            "autoencoder_split": LightningAutoencoder,
            "direct": Direct
            }
        
        # Define required parameters and their valid values
        required = {"arch_type": ["autoencoder_joint", "autoencoder_split", "direct"]}  # Require arch_type parameter and ensure it's a string
        parameterInterpreter = ParameterInterpreter(interpretation, requiredParams=required)

        # Check required parameters
        parameterInterpreter.checkRequiredParams(params)
        arch_type = params["arch_type"]
        arch = parameterInterpreter.interpret(arch_type)
        
        # Instantiate the architecture with the given parameters
        return arch(params)
        
    def optimize(self):
        """
        Optimize the architecture using the provided optimizer and configuration.
        Returns:
            L.LightningModule: The optimized architecture.
        """
        return self.optimizer.optimize(self.build_architecture, self.config)