import torch
import torch.nn as nn
import lightning as L
import optuna
import yaml
from . import LTorch

# implementation for a single architecture in the file

class Pipeline:
    # reads and stores architecture
    def __init__(self, train_data, val_data, test_data, path_arch: str, path_hyper: str):
        pass

    def optimize(self):
        pass

    def objective(self, trial:optuna.trial.BaseTrial):
        pass