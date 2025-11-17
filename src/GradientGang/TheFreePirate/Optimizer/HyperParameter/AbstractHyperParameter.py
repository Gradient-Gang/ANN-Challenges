import optuna

class AbstractHyperparameter:
    def getValue(self, trial: optuna.Trial, caller):
        pass

class AbstractValueHyperparameter (AbstractHyperparameter):
    def getValue(self, trial, caller):
        pass