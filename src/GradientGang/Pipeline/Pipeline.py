"""
hyper.yaml

"param1":
    "type": "categ" | "float" | "int"
    ...
"""

import pytorch_lightning as L
from .Architectures.LightningAutoencoder import LightningAutoencoder
from .Architectures.Direct import Direct
from .Utils.ParameterInterpreter import ParameterInterpreter
from .DataLoader.DataLoader import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class Pipeline:
    def __init__(
        self, 
        dict_loader_config: dict
    ):
        
        """
        Initializes the Pipeline with dataset, test data, optimizer, and configuration.
        Args:
            dict_config (dict, optional): Configuration dictionary.
        """

        # Load configuration
        self.data_loader = DataModule(dict_loader_config)

    def build_architecture(self, params) -> L.LightningModule:
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
    
    # TODO: CV can be added here
    def fit_and_validate(self, dict_arch, dict_data):
        # callbacks setup
        early_stopping = EarlyStopping(monitor="val_F1", patience=10, mode="max")
        checkpoint_callback = ModelCheckpoint(monitor="val_F1", mode="max")

        # Build architecture
        arch: L.LightningModule = self.build_architecture(dict_arch)

        # Configure loader
        self.data_loader.setup(**dict_data)

        # Train the architecture
        trainer: L.Trainer = L.Trainer(callbacks=[checkpoint_callback, early_stopping], max_epochs=5)
        trainer.fit(arch, train_dataloaders=self.data_loader.train_dataloader(), val_dataloaders=self.data_loader.val_dataloader())

        return checkpoint_callback.best_model_score