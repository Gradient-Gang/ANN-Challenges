import abc
import lightning as L


class AbstractLTorch (abc.ABC, L.LightningModule):
    @abc.abstractmethod
    def __init__(self, architecture: dict):
        pass

    @abc.abstractmethod
    def __init__(self, architecture_path: str):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        pass
    
    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        pass