import optuna
from .AbstractHyperParameter import AbstractHyperparameter, AbstractValueHyperparameter

class FloatHyperparameter (AbstractHyperparameter):
    def __init__(self, name, low, high, step=None, log=False):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def getValue(self, trial: optuna.Trial, caller):
        return trial.suggest_float(name=self.name, low=self.low, high=self.high, log=self.log, step=self.step)
    
class IntHyperparameter (AbstractHyperparameter):
    def __init__(self, name, low, high, step=1, log=False):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def getValue(self, trial: optuna.Trial, caller):
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, log=self.log, step=self.step)
    
class CategoricalHyperparameter (AbstractHyperparameter):
    def __init__(self, name, categories: list):
        self.name = name
        self.categories = categories

    def getValue(self, trial: optuna.Trial, caller):
        return trial.suggest_categorical(name=self.name, choices=self.categories)

# Global hyperparameters
"""
    'globalFeaturesEncoderNumLayers': 2,
    'globalFeaturesEmbeddingDim': 64,
    'globalFeaturesDropout': 0.5,
"""
class GlobalHyperparameter (AbstractHyperparameter):
    def __init__(self, hyperparameter: AbstractHyperparameter, values: list[AbstractHyperparameter]):
        self.hyperparameter = hyperparameter
        self.children = values
        self.val = None
        self.call_counter = len(values) + 1

        for h in self.children:
            h.parent = self

    def getValue(self, trial, caller):
        if self.call_counter == len(self.children) + 1:
            self.val = self.hyperparameter.getValue(trial, self)
        self.call_counter -= 1

        if issubclass(type(caller), AbstractValueHyperparameter):
            if self.val == True:
                caller.value = caller.hyperparam.getValue(trial, self)

        if self.call_counter == 0:
            self.call_counter = len(self.children)
            
        return self.val
            

class GlobalValuesHyperparameter (AbstractValueHyperparameter):
    def __init__(self, hyperparam: AbstractHyperparameter):
        self.hyperparam = hyperparam
        self.value = None

    def getValue(self, trial, caller):
        self.parent.getValue(trial, self)
        
        return self.value