import optuna
from ..Utils.ParameterInterpreter import ParameterInterpreter
from ..Pipeline import Pipeline

"""
Structure:
- dataloader
- hyper_dataloader
    - HYPERPARAMETER_DATA
- arch
- hyper_arch
    - HYPERPARAMETER_DATA

Definition of HYPERPARAMETER_DATA:
- name
- type one of {categ, float, int, categ_arch, repeated_arch}
- opts
    - categ {choices*: list}
    - float {min*, max*, step, log} all data is float, log is bool
    - int {min*, max*, step, log} all data is int, log is bool
    - arch {choices*: list, min*, max*}
"""

class OptunaOptimizer:
    # setup section
    class HyperParameter:
        def __init__(self, params):
            self.name = params["name"]
            self.type = params["type"]
            self.opts = params["opts"]
            self.paths = params["paths"]
        
        def getValue(self, trial: optuna.trial.BaseTrial):
            if self.type == "categ":
                return trial.suggest_categorical(self.name, **self.opts)
            elif self.type == "float":
                return trial.suggest_float(self.name, **self.opts)
            elif self.type == "int":
                return trial.suggest_int(self.name, **self.opts)
            elif self.type == "value":
                return self.opts["value"]
            elif self.type == "arch":
                raise NotImplementedError("Architecture and layer suggestion not implemented yet")

    def __init__(self, dict_config):
        # setup hyperparameters
        self.hyperparams_data = self.load_hyperparameters(dict_config["hyper_dataloader"])
        self.hyperparams_arch = self.load_hyperparameters(dict_config["hyper_arch"])

        # setup structure
        self.dataloader = dict_config["dataloader"]
        self.architecture = dict_config["arch"]

        # setup pipeline
        self.pipeline = Pipeline(dict_config["dataloader"])

    def load_hyperparameters(self, list_hyperparams: list):
        hp = {}

        for h in list_hyperparams:
            hp[h] = OptunaOptimizer.HyperParameter(list_hyperparams[h])
        
        return hp

    # def build_architecture_skeleton(self, arch, path):
    #     sk = None

    #     if isinstance(arch, list):
    #         sk = []
    #         for i in range(len(arch)):
    #             sk.append(self.build_architecture_skeleton(arch[i], path + [i]))
    #     elif isinstance(arch, dict):
    #         sk = {}
    #         for k in arch:
    #             if k in self.hyperparams_arch:   # hyper parameter found
    #                 self.hyperparams_arch[k].path.append(path)
    #             elif isinstance(arch[k], dict) or isinstance(arch[k], list):
    #                 sk[k] = self.build_architecture_skeleton(arch[k], path + [k])
    #             else:
    #                 sk[k] = arch[k]
    #     else:
    #         raise ValueError("build_architecture_skeleton called on neither list or dictionary")
    #     return sk
    
    # objective section
    def objective(self, trial: optuna.trial.BaseTrial):
        dict_arch = self.build_architecture_data(self.dict_arch, trial)
        dict_data = self.getParams(self.dict_data_d, trial)

        return self.pipeline.fit_and_validate(dict_arch, dict_data)

    def optimize(
        self,
        n_trials: int = None
    ) -> optuna.study:
        """
        Optimize the architecture builder function using Optuna.
        Args:
            architecture_builder (types.FunctionType): The function that builds the architecture.
            params (dict): The parameter configuration dictionary.
            n_trials (int, optional): The number of trials to run. Defaults to None.
        Returns:
            optuna.study: The Optuna study object containing the optimization results.
        """

        # Create and run the Optuna study
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(self.objective, n_trials=n_trials)

        # Return the completed study
        return study
