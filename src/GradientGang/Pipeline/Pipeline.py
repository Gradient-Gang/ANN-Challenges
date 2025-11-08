"""
hyper.yaml

"param1":
    "type": "categ" | "float" | "int"
    ...
"""

import lightning as L
from ..Optimizer.Optimizer import Optimizer 
import yaml

class Pipeline:
    def __init__(
                 self, 
                 dataset: L.LightningDataModule, 
                 test: L.LightningDataModule, 
                 optimizer: Optimizer, 
                 dict_config: dict = None, 
                 path_config: str = None):
        if dict_config == path_config == None or (dict_config != None and path_config != None):
            raise ValueError("dict_config or path_config have to be assigned")
        elif not(isinstance(dataset, L.LightningDataModule) and isinstance(test, L.LightningDataModule)):
            raise ValueError("dataset and test must be L.LightningDataModule")
        elif dict_config is not None and not isinstance(dict_config, dict):
            raise ValueError("dict_config must be a dictionary")
        elif path_config is not None and not isinstance(path_config, str):
            raise ValueError("path_config must be a string")

        self.dataset: L.LightningDataModule = dataset
        self.test: L.LightningDataModule = test
        self.optimizer: Optimizer = optimizer

        if dict_config is not None:
            self.config = dict_config
        elif path_config is not None:
            with open(path_config, "r") as f:
                self.config = yaml.safe_load(f)

    def build_architecture(self, params: dict) -> L.LightningModule:
        arch = None

        if params["arch_type"] == "autoencoder_split":
            pass
        elif params["arch_type"] == "autoencoder_joint":
            pass
        elif params["arch_type"] == "direct":
            pass
        else:
            raise ValueError("Invalid architecture type")
        
        return arch
        
    def optimize(self):
        return self.optimizer.optimize(self.build_architecture, self.config)