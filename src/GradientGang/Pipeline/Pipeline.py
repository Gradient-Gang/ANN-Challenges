"""
Pipeline:
1) Define configuration in a .yaml file,
    it should define all the hyperparameters for the Pytorch Lightning model
2) Write the objective function in Optuna

File structure:
- architecture1
    name: "string"
    components
        -
            name: "comp1"
            parameters:
            -
                name: "param1"
                type: "range" OR "value" OR "class"
                params:
                    min: MIN
                    max: MAX
                params: CONST
                params:
                    - c1
                    - c2
                    - ...
            -
                name: "param2"
                ...
        -
            name: "comp2"
    loss
    n_trials
- architecture2
"""

import torch
import torch.nn as nn
import lightning as L
import optuna
import yaml
from . import LTorch

# implementation for a single architecture in the file

class Pipeline:
    # reads and stores architecture
    def __init__(self, train, val, test, configuration_file: str, componentsDict: dict):
        self.train = train
        self.val = val
        self.test = test
        
        self.componentsDict = componentsDict
        
        with open(configuration_file, "r") as f:
            self.architecture = yaml.safe_load(f)

        self.components = self.architecture["components"]
        self.loss = self.architecture["loss"]
        self.n_trials = self.architecture["n_trials"]

        for c in self.components:
            c["values"] = {}
            for p in c["parameters"]:
                c["values"][p["name"]] = None

    def optimize(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.n_trials)

        return study.best_params

    def getSuggestion(self, name, type, params, trial:optuna.trial.BaseTrial):
        if type == "range":
            return trial.suggest_uniform(name, params["min"], params["max"])
        elif type == "value":
            return params
        elif type == "class":
            return trial.suggest_categorical(name, params)

    def objective(self, trial:optuna.trial.BaseTrial):
        # choose hyperparameters
        for c in self.components:
            for p in c["parameters"]:
                c["values"][p["name"]] = self.getSuggestion(p["name"], p["type"], p["params"], trial)

        # assemble architecture
        model = LTorch.LTorch(self.components, self.componentsDict, self.loss)
        trainer = L.Trainer()
        trainer.fit(model, train_dataloaders=self.train, val_dataloaders=self.val)
        # evaluate
        return trainer.validate()