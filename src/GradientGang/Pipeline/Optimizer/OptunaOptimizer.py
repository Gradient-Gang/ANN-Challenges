import optuna
from ..Utils.ParameterInterpreter import ParameterInterpreter
from ..Pipeline import Pipeline
from .HyperParameter import HyperParameter, HyperParameterConstraint

"""
Structure:
- dataloader
- hyper_dataloader
    - HYPERPARAMETER_DATA
- constr_dataloader
    - CONSTR_DATA
- arch
- hyper_arch
    - HYPERPARAMETER_DATA
- constr_arch
    - CONSTR_DATA

Definition of HYPERPARAMETER_DATA:
- name
- type one of {categ, float, int, arch}
- opts
    - categ {choices*: list}
    - float {min*, max*, step, log} all data is float, log is bool
    - int {min*, max*, step, log} all data is int, log is bool
    - arch {names*: list, archs*: dict, min_layers*, max_layers*, constraint}
- path

Definition of CONSTRAINTS:
- hyperparams: list
- function
"""

class OptunaOptimizer:
    # setup section
    def __init__(self, dict_config):
        self.constr_map = {}

        # setup hyperparameters
        self.hyperparams_dataloader, self.constr_dataloader = self.load_hyperparameters(
            dict_config["hyper_dataloader"],
            dict_config["constr_dataloader"]
        )
        self.hyperparams_arch, self.constr_arch = self.load_hyperparameters(
            dict_config["hyper_architecture"],
            dict_config["constr_architecture"]
        )

        # setup structure
        self.architecture = dict_config["architecture"]

        # setup pipeline
        self.pipeline = Pipeline(dict_config["dataloader"])

    def load_hyperparameters(self, list_hyperparams: list, constraints: list):
        hp = {}
        constr = {}

        # build all hyperparameters (computes also nested ones in the HyperParameter constructor)
        for h in list_hyperparams:
            hp[h["name"]] = HyperParameter(h, constr)
        
        # assigns all constraints
        for c in constraints:
            constr[c["name"]] = HyperParameterConstraint(self.constr_map[c["function"]])

            for h in c["hyperparams"]:
                constr[c["name"]].params.append(hp[h])

        return (hp, constr)
    
    # objective section
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

    def objective(self, trial: optuna.trial.BaseTrial):
        dict_data = self.build({}, self.hyperparams_dataloader, trial)
        dict_arch = self.build(self.architecture, self.hyperparams_arch, trial)

        return self.pipeline.fit_and_validate(dict_arch, dict_data)

    def build(self, skeleton: dict, hyperparams: dict, trial: optuna.trial.BaseTrial):
        arch = skeleton.copy()

        for h in hyperparams.values():
            curr = arch
            for k in h.path[:-1]:
                curr = curr[k]
            if isinstance(curr, list):
                curr[h.path[-1]:h.path[-1]] = h.getValue(trial)
            else:
                curr[h.path[-1]] = h.getValue(trial)
        
        return arch