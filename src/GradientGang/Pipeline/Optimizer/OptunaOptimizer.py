import optuna
from ..Utils.ParameterInterpreter import ParameterInterpreter
from ..Pipeline import Pipeline

class OptunaOptimizer:
    def __init__(self, dict_config):
        self.dict_data_s = dict_config["data_static"]
        self.dict_data_d = dict_config["data_dynamic"]
        self.dict_arch = dict_config["arch"]

        self.pipeline = Pipeline(self.dict_data_s)

    def getParams(
        self, params: dict, trial: optuna.trial.BaseTrial
    ):
        """
        Get the parameters for the current trial.
        Args:
            params (dict): The parameter configuration dictionary.
            trial (optuna.trial.BaseTrial): The current Optuna trial.
        Returns:
            dict: The interpreted parameters for the trial.
        """

        # Dictionary to hold the interpreted parameter values
        vals = {}

        # Mapping of parameter types to Optuna suggestion methods
        interpretation = {
            "categ": trial.suggest_categorical,
            "float": trial.suggest_float,
            "int": trial.suggest_int,
            "value": lambda x: x
        }
        
        # Required parameters for each parameter type
        required = {"type": str, "params": dict}
        parameterInterpreter = ParameterInterpreter(interpretation, requiredParams=required)

        # Interpret each parameter using the ParameterInterpreter
        for k in params:
            parameterInterpreter.checkRequiredParams(params[k])

            vals[k] = parameterInterpreter.interpret(params[k]["type"])(**params[k]["params"])

        return vals

    def objective(self, trial: optuna.trial.BaseTrial):
        dict_arch = self.getParams(self.dict_arch, trial)
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