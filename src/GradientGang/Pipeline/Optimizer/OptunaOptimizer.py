import optuna
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import types
from . import Optimizer
from ..Utils.ParameterInterpreter import ParameterInterpreter

class OptunaOptimizer (Optimizer.Optimizer):

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
            "int": trial.suggest_int}
        
        # Required parameters for each parameter type
        required = {"type": str, "params": dict}
        parameterInterpreter = ParameterInterpreter(interpretation, requiredParams=required)

        # Interpret each parameter using the ParameterInterpreter
        for k in params:
            parameterInterpreter.checkRequiredParams(params[k])

            vals[k] = parameterInterpreter.interpret(params[k]["type"])(**params[k]["params"])

        return vals

    def optimize(
        self, 
        architecture_builder: types.FunctionType,
        params: dict,
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

        # Store the architecture builder and parameters for use in the objective function
        self.build_architecture = architecture_builder
        self.params = params

        # Create and run the Optuna study
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(self.objective, n_trials=n_trials)

        # Return the completed study
        return study

    def objective(
        self, trial: optuna.trial.BaseTrial
    ):
        """
        Objective function for the Optuna study.

        Args:
            trial (optuna.trial.BaseTrial): The current Optuna trial.

        Returns:
            float: The objective value to be minimized or maximized.
        """

        # Get the parameters for the current trial
        params = self.getParams(self.params, trial)

        # Build the architecture with the current parameters
        arch: L.LightningModule = self.build_architecture(params)

        # Set up early stopping and model checkpointing
        early_stopping = EarlyStopping(monitor="val_f1", patience=10, mode="max")
        checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max")

        # Train the architecture
        trainer: L.Trainer = L.Trainer(
            callbacks=[checkpoint_callback, early_stopping],
            max_epochs=100,
            log_every_n_steps=1,
            enable_progress_bar=False
        )
        trainer.fit(arch)

        # Return the best model score from the checkpoint
        return checkpoint_callback.best_model_score