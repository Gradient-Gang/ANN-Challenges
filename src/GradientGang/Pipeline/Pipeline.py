"""
Pipeline:
1) Define configuration in a .yaml file,
    it should define all the hyperparameters for the Pytorch Lightning model
2) Write the objective function in Optuna
"""

import torch
import torch.nn as nn
import lightning as L
import optuna
import yaml

class Pipeline:
    pass