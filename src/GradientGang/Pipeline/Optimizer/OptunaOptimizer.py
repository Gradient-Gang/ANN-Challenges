import optuna
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
import types
from . import Optimizer
from ..Utils.ParameterInterpreter import ParameterInterpreter

class OptunaOptimizer (Optimizer.Optimizer):
    def getParams(self, params: dict, trial: optuna.trial.BaseTrial):
        vals = {}

        interpretation = {
            "categ": trial.suggest_categorical,
            "float": trial.suggest_float,
            "int": trial.suggest_int}
        
        required = {"type": str, "params": dict}
        
        parameterInterpreter = ParameterInterpreter(interpretation, requiredParams=required)

        for k in params:
            parameterInterpreter.checkRequiredParams(params[k])

            vals[k] = parameterInterpreter.interpret(params[k]["type"])(**params[k]["params"])

        return vals

    def optimize(self, 
                architecture_builder: types.FunctionType,
                params: dict,
                train_data: L.LightningDataModule,
                val_data: L.LightningDataModule,
                n_trials: int = None) -> optuna.study:
        self.build_architecture = architecture_builder
        self.params = params

        self.train_data = train_data
        self.val_data = val_data

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(self.objective, n_trials=n_trials)

        return study

    def objective(self, trial: optuna.trial.BaseTrial):
        params = self.getParams(self.params, trial)

        arch: L.LightningModule = self.build_architecture(params)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',  # since you want to maximize F1
            save_top_k=1
        )

        trainer: L.Trainer = L.Trainer(callbacks=[checkpoint_callback])
        trainer.fit(arch, self.train_data, self.val_data)

        return checkpoint_callback.best_model_score