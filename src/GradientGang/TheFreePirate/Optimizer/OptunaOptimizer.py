import optuna
import numpy as np
from ..Piepeline.PiratePipeline import PiratePipeline
from .HyperParameter.AbstractHyperParameter import AbstractHyperparameter

from ..Utils import getRaise

"""
phi = {
    'dataParams': {
        'numFolds': 5,
        'includeTestInFolds': True,
        'augmentTestSet': True,
        'dataAugmentationParams':{
            'nCopies': 2,
            'keepOriginal': True,
            'scaleRange': 0.1,
            'jitterStdDev': 0.05,
            'offsetRange': 0.1,
            'maxWarpFraction': 0.2,
            'windowSize': 10,
            'windowStride': 5,
        }
    },
    'modelParams': {
        'f1AverageStrategy': 'weighted',
        'numClasses': 3,
        'reconstructionLossWeight': 0.5,
        'useGlobalFeatures': False,
        'learningRate': 1e-3,
        'weightDecay': 1e-2,
        'activationFunction': 'relu',
        'timeSeriesEncoderNumLayers': 2,
        'timeSeriesEmbeddingDim': 64,
        'timeSeriesDropout': 0.2,
        'predictorNumLayers': 2,
        'predictorDropout': 0.1,
    }
}
"""

class OptunaOptimizer:
    database = "REMOVED_KEY"

    class MockTrial:
        def __init__(self, params, study):
            self.params = params
            self.study = study
            self.storage = study._storage  # Access the actual storage
            self._trial_id = study.best_trial._trial_id  # Use best trial's ID

        def suggest_int(self, name, low, high, *, step: int, log = False):
            return getRaise(self.params, name, "best_params in MockTrial")
        
        def suggest_categorical(self, name, choices):
            return getRaise(self.params, name, "best_params in MockTrial")
        
        def suggest_float(self, name, low, high, *, step = None, log = False):
            return getRaise(self.params, name, "best_params in MockTrial")    

    # setup section
    def __init__(self, name: str, dataParams, callbackParams, params):
        self.params = params

        #self.study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), storage=OptunaOptimizer.database)
        self.study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), storage=None)
        self.pipeline = PiratePipeline(name, dataParams, callbackParams, seed = 42)
    
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
        self.study.optimize(self.objective, n_trials=n_trials)

        # Return the completed study
        return self.study

    def objective(self, trial: optuna.trial.BaseTrial):
        self.current_trial = trial

        self.phi = self._build_arch(self.params)

        self.pipeline.setupKfoldEvaluation(self.phi)
        results = self.pipeline.kFoldEvaluation(self.phi, [self._optuna_callback])

        self.current_trial.set_user_attr("mean", np.mean(results))
        self.current_trial.set_user_attr("std", np.std(results))

        return np.mean(results)

    def _optuna_callback(self, results):
        fold_number = len(results)
        self.current_trial.report(np.mean(results), fold_number)
        self.current_trial.set_user_attr(f"fold_{fold_number}", results[-1])

    def _build_arch(self, params: dict):
        phi = {}

        for (key, value) in params.items():
            if isinstance(value, dict):
                phi[key] = self._build_arch(value)
            elif issubclass(type(value), AbstractHyperparameter):
                phi[key] = value.getValue(self.current_trial, self)
            else:
                phi[key] = value
        
        return phi
    
    def getBestModel(self):
        self.current_trial = OptunaOptimizer.MockTrial(self.study.best_params, self.study)

        return self._build_arch(self.params)