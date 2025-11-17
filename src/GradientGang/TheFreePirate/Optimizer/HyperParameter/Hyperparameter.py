import optuna
from .AbstractHyperParameter import AbstractHyperParameter

class FloatHyperparameter (AbstractHyperParameter):
    def __init__(self, name, low, high, step=None, log=False):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def getValue(self, trial: optuna.Trial):
        return trial.suggest_float(name=self.name, low=self.low, high=self.high, log=self.log, step=self.step)
    
class IntHyperparameter (AbstractHyperParameter):
    def __init__(self, name, low, high, step=1, log=False):
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def getValue(self, trial: optuna.Trial):
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, log=self.log, step=self.step)
    
class CategoricalHyperparameter (AbstractHyperParameter):
    def __init__(self, name, categories: list):
        self.name = name
        self.categories = categories

    def getValue(self, trial: optuna.Trial):
        return trial.suggest_categorical(name=self.name, choices=self.categories)

# class Global(AbstractHyperParameter):
#     def __init__(self, value):
#         self.value = value

#     def getValue(self, trial):
#         return self.value

# class FixRelated(AbstractHyperParameter):
#     def __init__(self, related: Related):
#         self.related = related
    
#     def getValue(self, trial):
#         trial.suggest_categorical

#         if 
#             self.related.value = 1
#         return 2