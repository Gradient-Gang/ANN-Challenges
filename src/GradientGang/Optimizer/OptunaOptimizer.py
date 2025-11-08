import optuna
import lightning as L
import types
from . import Optimizer

class OptunaOptimizer (Optimizer.Optimizer):
    def getParams(self, params: dict, trial: optuna.trial.BaseTrial):
        vals = {}
        
        for k in params:
            if params[k]["type"] == "categ":        # params: list of categories
                vals[k] = trial.suggest_categorical(k, params[k]["seq"])
            elif params[k]["type"] == "float":      # params: low, high, step (optional), log
                kwargs = {"log": params[k]["log"]}
                if "step" in params[k]:
                    kwargs["step"] = params[k]["step"]
                vals[k] = trial.suggest_float(k, params[k]["low"], params[k]["high"], **kwargs)
            elif params[k]["type"] == "int":        # params: low, high, step (optional), log
                kwargs = {"log": params[k]["log"]}
                if "step" in params[k]:
                    kwargs["step"] = params[k]["step"]
                vals[k] = trial.suggest_int(k, params[k]["low"], params[k]["high"], **kwargs)
            elif params[k]["type"] == "value":
                vals[k] = params[k]["value"]
            else:
                raise TypeError("Pipeline.getParams: Invalid type for parameter {} ({})".format(k, params[k]["type"]))
    
        return vals

    def optimize(self, architecture_builder: types.FunctionType, params: dict, n_trials: int = 10) -> optuna.study:
        self.build_architecture = architecture_builder
        self.params = params

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(self.objective, n_trials=n_trials)

        return study

    def objective(self, trial: optuna.trial.BaseTrial):
        params = self.getParams(self.params, trial)

        arch: L.LightningModule = self.build_architecture(params)

        return arch.validation_step()