from ..DataHandling.PirateDataModule import PirateDataModule
from ..DataHandling.DataAugmentation.TensorAugmenter import (
    TensorAugmenter,
    ScaleAugmenter,
    JitterAugmenter,
    OffsetAugmenter,
    TimeWarpAugmenter,
    SequentialAugmenter,
    WindowingAugmenter,
)
import os
from ..Architectures.PirateLightningModule import PirateLightningModule
from ..Utils import getRaise
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

from copy import deepcopy

from typing import Callable

from ..Architectures.AggregationStrategies import MajorityVotingAggregationStrategy

import torch

import pandas as pd


class PiratePipeline:
    def __init__(
        self,
        name: str,
        dataParams: dict,
        callbacksParams: dict,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        self.name = name

        self.seedEverything(seed)

        self.dataModule = PirateDataModule.fromCSV(**dataParams)

        self.callbacksParams = callbacksParams

    def seedEverything(self, seed: int):
        pl.seed_everything(seed)
        self.seed = seed

    def setupKfoldEvaluation(self, params: dict):
        paramName = "PiratePipeline.setupKfoldEvaluation"
        dataParams = getRaise(params, "dataParams", paramName)
        modelParams = getRaise(params, "modelParams", paramName)

        self.dataSetup(dataParams)

        self.modelParams = deepcopy(modelParams)
        self.modelParams["dataModuleInfo"] = self.dataModule.getDataInfoKFold()

    def evaluateParameters(self, params: dict) -> float:

        return 0.0

    def dataSetup(self, dataParams: dict, *args, **kwargs):
        dictionaryName = "dataParams in PiratePipeline.dataSetup"
        numFold = getRaise(dataParams, "numFolds", dictionaryName)
        includeTestInFolds = getRaise(dataParams, "includeTestInFolds", dictionaryName)
        augmentTestSet = getRaise(dataParams, "augmentTestSet", dictionaryName)

        # Set up augmentations
        augmentationParams = getRaise(
            dataParams, "dataAugmentationParams", dictionaryName
        )
        augmentationParamsName = "dataAugmentationParams in PiratePipeline.dataSetup"
        nCopies = getRaise(augmentationParams, "nCopies", augmentationParamsName)
        keepOriginal = getRaise(
            augmentationParams, "keepOriginal", augmentationParamsName
        )
        scaleRange = getRaise(augmentationParams, "scaleRange", augmentationParamsName)
        jitterStdDev = getRaise(
            augmentationParams, "jitterStdDev", augmentationParamsName
        )
        offsetRange = getRaise(
            augmentationParams, "offsetRange", augmentationParamsName
        )
        maxWarpFraction = getRaise(
            augmentationParams, "maxWarpFraction", augmentationParamsName
        )

        # Set up Sequential Augmenter
        augmenters: list[TensorAugmenter] = [
            ScaleAugmenter(rangeScale=scaleRange),
            JitterAugmenter(jitterStdDev=jitterStdDev),
            OffsetAugmenter(rangeOffset=offsetRange),
            TimeWarpAugmenter(maxWarpFraction=maxWarpFraction),
        ]

        augmenter = SequentialAugmenter(
            nCopies=nCopies,
            keepOriginal=keepOriginal,
            augmenters=augmenters,
        )

        # Set up Windowing
        windowSize = getRaise(augmentationParams, "windowSize", augmentationParamsName)
        windowStride = getRaise(
            augmentationParams, "windowStride", augmentationParamsName
        )

        windowingAugmenter = WindowingAugmenter(
            windowSize=windowSize,
            stride=windowStride,
        )

        # Setup K-Folds with augmentations
        self.dataModule.setupKFolds(
            nFolds=numFold,
            dataAugmenter=augmenter,
            windowingAugmenter=windowingAugmenter,
            includeTestInFolds=includeTestInFolds,
            augmentTestSet=augmentTestSet,
        )

    def modelSetup(self, modelParams: dict, *args, **kwargs):
        self.model = PirateLightningModule(modelParams)

    def setupCallbacks(
        self, foldIndex: int
    ) -> tuple[list[pl.Callback], TensorBoardLogger]:
        paramName = "callbackParam is  PiratePipeline.setupCallbacks"
        callbacksList: list[pl.Callback] = []

        baseLogDir = getRaise(self.callbacksParams, "baseLogDir", paramName)
        baseLogDir = os.path.join(baseLogDir, self.name)

        # Tensorboard logger
        loggerCallback = TensorBoardLogger(
            save_dir=baseLogDir,
            name=f"fold_{foldIndex}",
            version=None,
        )

        # Checkpoint callback
        checkpointDir = os.path.join(loggerCallback.log_dir, "checkpoints")
        self.checkpointCallback = ModelCheckpoint(
            dirpath=checkpointDir,
            filename="trial-{epoch:02d}-val_F1={val/F1:.3f}",
            monitor="val/f1_score",
            mode="max",
            save_top_k=1,
            verbose=False,
        )
        callbacksList.append(self.checkpointCallback)

        # SWA callback
        swaCallback = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacksList.append(swaCallback)

        # Early stopping callback
        earlyStoppingPatience = getRaise(
            self.callbacksParams, "earlyStoppingPatience", paramName
        )
        earlyStoppingCallback = EarlyStopping(
            monitor="val/f1_score",
            patience=earlyStoppingPatience,
            mode="max",
            verbose=False,
        )
        callbacksList.append(earlyStoppingCallback)

        return callbacksList, loggerCallback

    def evaluateOnFold(self, foldIndex: int) -> float:
        # Model setup
        self.modelSetup(self.modelParams)

        # Extract data
        trainDataLoader, valDataLoader = self.dataModule.getFoldDataLoaders(foldIndex)

        # Setup trainer
        callbacksList, loggerCallback = self.setupCallbacks(foldIndex)

        # SetupTrainer
        maxEpochs = getRaise(
            self.callbacksParams, "maxEpochs", "callbacksParams in PiratePipeline"
        )
        trainer = pl.Trainer(
            callbacks=callbacksList,
            logger=loggerCallback,
            enable_progress_bar=True,
            enable_model_summary=False,
            max_epochs=maxEpochs,
        )

        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=trainDataLoader,
            val_dataloaders=valDataLoader,
        )

        # Load best checkpoint
        bestModelPath = self.checkpointCallback.best_model_path
        self.model = PirateLightningModule.load_from_checkpoint(bestModelPath)

        # Evaluate on validation set
        valMetrics = trainer.validate(
            self.model,
            dataloaders=valDataLoader,
            verbose=False,
        )
        valF1Score = valMetrics[0]["val/f1_score"]
        self.bestModels.append(self.model)
        return valF1Score

    def kFoldEvaluation(
        self,
        params: dict,
        foldResultCallbacks: list[Callable[[list[float]], None]] = [],
    ) -> list[float]:
        self.setupKfoldEvaluation(params)

        numFolds = self.dataModule.getDataInfoKFold()["numFolds"]
        evaluations = []

        self.bestModels = []

        for foldIndex in range(numFolds):
            print(f"\n=== Evaluating Fold {foldIndex + 1}/{numFolds} ===")
            foldEvaluation = self.evaluateOnFold(foldIndex)
            print(f"Fold {foldIndex} Evaluation: {foldEvaluation:.4f}")

            evaluations.append(foldEvaluation)

            # Call fold result callbacks
            for callback in foldResultCallbacks:
                callback(evaluations)

        return evaluations

    def predictFromEnsembledModel(self):
        if not self.bestModels:
            raise ValueError("No trained models available for prediction.")

        aggregationStrategy = MajorityVotingAggregationStrategy()

        kFoldPredictions = torch.tensor([])
        for model in self.bestModels:
            model.eval()
            testLoader = self.dataModule.getTestLoader()
            all_predictions = torch.tensor([])
            with torch.no_grad():
                for batch in testLoader:
                    if isinstance(batch, (list, tuple)):
                        features = batch[0]
                    else:
                        features = batch

                    predictions, _, _ = model(features)
                    all_predictions = torch.cat((all_predictions, predictions), dim=0)
            kFoldPredictions = torch.cat(
                (kFoldPredictions, all_predictions.unsqueeze(0)), dim=0
            )

        kFoldPredictions = kFoldPredictions.permute(
            1, 0, 2
        )  # (nSamples, nFolds, nClasses)
        finalPredictions = aggregationStrategy.aggregate(kFoldPredictions)

        return finalPredictions
