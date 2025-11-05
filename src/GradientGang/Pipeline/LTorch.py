import torch
import torch.nn as nn
import lightning as L

class LTorch (L.LightningModule):
    def __init__(self, components: list, componentsDict: dict):
        super().__init__()

        self.arch = nn.Sequential()

        for c in components:
            self.arch.append(componentsDict[c[0]](**(c[1])))   # parameters are a dictionary decompressed
    
    def forward(self, x):
        return self.arch.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.arch(x)

        return self.loss(y_pred, y)