import optuna

class AbstractHyperParameter:
    def getValue(self, trial: optuna.Trial):
        pass