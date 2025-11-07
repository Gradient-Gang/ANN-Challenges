"""
hyper.yaml

"param1":
    "type": "categ" | "float" | "int"
    ...
"""

import abc
import torch
import torch.nn as nn
import lightning as L
import optuna
import yaml
from . import LTorch

class Pipeline (abc.ABC):
    def __init__(self, train_data, val_data, test_data, path_arch: str, path_hyper: str):
        self.arch = yaml.safe_load(path_arch)
        self.hyper = yaml.safe_load(path_hyper)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    # might build from arch, or restart from checkpoint?
    @abc.abstractmethod
    def build_architecture(self, arch: dict, params: dict) -> LTorch.AbstractLTorch:
        pass

    def getParams(self, params: dict, trial: optuna.trial.BaseTrial):
        vals = {}
        
        for k in params:
            if params[k]["type"] == "categ":        # params: list of categories
                vals[k] = trial.suggest_categorical(k, params[k]["seq"])
            elif params[k]["type"] == "float":      # params: low, high, step, log
                vals[k] = trial.suggest_float(k, params[k]["low"], params[k]["high"], params[k]["step"], params[k]["log"])
            elif params[k]["type"] == "int":        # params: low, high, step, log
                vals[k] = trial.suggest_int(k, params[k]["low"], params[k]["high"], params[k]["step"], params[k]["log"])
            else:
                raise TypeError("Pipeline.getParams: Invalid type for parameter")
    
        return vals

    def optimize(self):
        study = optuna.create_study()
        study.optimize(self.objective)

        return study.best_params

    def objective(self, trial: optuna.trial.BaseTrial):
        params = self.getParams(self.hyper, trial)

        arch = self.build_architecture(self.arch, params)

        return arch.validation_step()