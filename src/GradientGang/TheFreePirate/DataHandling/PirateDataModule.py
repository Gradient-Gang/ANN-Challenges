import pytorch_lightning as lightning
import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    random_split,
    Subset,
)
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from sklearn.model_selection import StratifiedKFold
from .PirateDataset import PirateDataset
from .DataAugmentation.TensorAugmenter import (
    SequentialAugmenter,
    IdentityAugmenter,
    WindowingAugmenter,
)
import os


class PirateDataModule(lightning.LightningDataModule):
    @staticmethod
    def fromCSV(
        folderPath: str,
        trainTimeSeriesFileName: str,
        trainGlobalFeaturesFileName: str,
        trainLabelsFileName: str,
        testTimeSeriesFileName: str,
        testGlobalFeaturesFileName: str,
        *args,
        **kwargs,
    ):
        trainTimeSeriesDf = pd.read_csv(
            os.path.join(folderPath, trainTimeSeriesFileName)
        )
        trainGlobalFeaturesDf = pd.read_csv(
            os.path.join(folderPath, trainGlobalFeaturesFileName)
        )
        trainLabelsDf = pd.read_csv(os.path.join(folderPath, trainLabelsFileName))
        testTimeSeriesDf = pd.read_csv(os.path.join(folderPath, testTimeSeriesFileName))
        testGlobalFeaturesDf = pd.read_csv(
            os.path.join(folderPath, testGlobalFeaturesFileName)
        )

        return PirateDataModule(
            trainTimeSeriesDf=trainTimeSeriesDf,
            trainGlobalFeaturesDf=trainGlobalFeaturesDf,
            trainLabelsDf=trainLabelsDf,
            testTimeSeriesDf=testTimeSeriesDf,
            testGlobalFeaturesDf=testGlobalFeaturesDf,
            *args,
            **kwargs,
        )

    def __init__(
        self,
        trainTimeSeriesDf: pd.DataFrame,
        trainGlobalFeaturesDf: pd.DataFrame,
        trainLabelsDf: pd.DataFrame,
        testTimeSeriesDf: pd.DataFrame,
        testGlobalFeaturesDf: pd.DataFrame,
        labelsMapping: dict[str, int],  # Name -> index
        batch_size: int = 32,
        num_workers: int = 0,
        primaryKeyColumn: str = "sample_index",
        timeColumn: str = "time",
        labelsColumn: str = "label",
        columnsToIgnore: list[str] = ["joint_11"],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.primaryKeyColumn = primaryKeyColumn
        self.timeColumn = timeColumn
        self.columnsToIgnore = columnsToIgnore
        self.labelsMapping = labelsMapping
        self.labelsColumn = labelsColumn
        self.trainTimeSeriesDf = self._removeColumns(trainTimeSeriesDf)
        self.trainGlobalFeaturesDf = self._removeColumns(trainGlobalFeaturesDf)
        self.trainLabelsDf = self._mapLabels(trainLabelsDf)
        self.testTimeSeriesDf = self._removeColumns(testTimeSeriesDf)
        self.testGlobalFeaturesDf = self._removeColumns(testGlobalFeaturesDf)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._tensorize()

    def _removeColumns(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        return dataFrame.drop(columns=self.columnsToIgnore, errors="ignore")

    def _mapLabels(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        dataFrame = dataFrame.copy()
        dataFrame[self.labelsColumn] = dataFrame[self.labelsColumn].map(
            self.labelsMapping
        )
        return dataFrame

    def _getTimeSeriesTensor(self, timeSeriesDf: pd.DataFrame) -> torch.Tensor:
        pivot = timeSeriesDf.pivot_table(
            index=self.primaryKeyColumn, columns=self.timeColumn
        )

        columnsNames = pivot.columns.get_level_values(0).unique().to_list()
        sampleData = [pivot[colName].values for colName in columnsNames]

        timeSeriesTensor = torch.tensor(
            np.stack(sampleData, axis=-1), dtype=torch.float32
        )
        return timeSeriesTensor

    def _tensorize(self):
        # Convert training data to tensors
        self.trainTimeSeriesTensor = self._getTimeSeriesTensor(self.trainTimeSeriesDf)

        self.trainGlobalFeaturesTensor = torch.tensor(
            self.trainGlobalFeaturesDf.values, dtype=torch.float32
        )
        self.trainLabelsTensor = torch.tensor(
            self.trainLabelsDf.values, dtype=torch.long
        )[:, 1]

        # Convert test data to tensors
        self.testTimeSeriesTensor = self._getTimeSeriesTensor(self.testTimeSeriesDf)
        self.testGlobalFeaturesTensor = torch.tensor(
            self.testGlobalFeaturesDf.values, dtype=torch.float32
        )

    def getDataInfoKFold(self):
        if not hasattr(self, "folds"):
            raise ValueError("K-Folds not set up. Call setupKFolds() first.")

        return {
            "numFolds": len(self.folds),
            "foldSizes": [
                {
                    "trainSize": len(trainSubset),
                    "valSize": len(valSubset),
                }
                for trainSubset, valSubset in self.folds
            ],
            "shapes": {
                "train": self.folds[0][0].datasets[0].getShape(),
                "val": self.folds[0][1].getShape(),
            },
        }

    def setupKFolds(
        self,
        nFolds: int = 5,
        dataAugmenter: SequentialAugmenter = SequentialAugmenter(
            nCopies=0,
            keepOriginal=True,
            augmenters=[IdentityAugmenter()],
        ),
        windowingAugmenter: WindowingAugmenter = WindowingAugmenter(
            windowSize=160, stride=50
        ),
        includeTestInFolds: bool = True,
        augmentTestSet: bool = True,
    ):
        print(
            f"[PirateDataModule] Setting up K-Folds({nFolds}) with data augmentation..."
        )

        # Prepare stratified K-Fold splits
        skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
        self.folds = []

        X_indices = np.arange(len(self.trainLabelsTensor))
        y_labels = self.trainLabelsTensor.numpy()

        if includeTestInFolds:
            if augmentTestSet:
                # Apply data augmentation to test data
                augmentedTestTimeSeries = dataAugmenter.augment(
                    self.testTimeSeriesTensor, dim=1
                )
                nCopies = dataAugmenter.nCopies + (
                    1 if dataAugmenter.keepOriginal else 0
                )
                # Repeat global features accordingly for test data
                repeatedTestGlobalFeatures = self.testGlobalFeaturesTensor.repeat(
                    nCopies, 1
                )

            else:
                nCopies = 1
                augmentedTestTimeSeries = self.testTimeSeriesTensor
                repeatedTestGlobalFeatures = self.testGlobalFeaturesTensor

            # Only apply windowing to test data
            augmentedTestTimeSeries = windowingAugmenter.augment(
                augmentedTestTimeSeries, dim=1
            )

            # Create unlabeled dataset from test data
            self.unlabeledDataset = PirateDataset(
                timeSeriesTensor=augmentedTestTimeSeries,
                globalFeaturesTensor=repeatedTestGlobalFeatures,
                labelsTensor=-torch.ones(
                    self.testTimeSeriesTensor.shape[0] * nCopies, dtype=torch.long
                ),  # dummy labels
            )

        for train_idx, val_idx in skf.split(X_indices, y_labels):
            # Apply data augmentation to training data
            augmentedTrainTimeSeries = dataAugmenter.augment(
                self.trainTimeSeriesTensor[train_idx], dim=1
            )
            augmentedTrainTimeSeries = windowingAugmenter.augment(
                augmentedTrainTimeSeries, dim=1
            )

            # Repeat global features and labels accordingly for training data
            nCopies = dataAugmenter.nCopies + (1 if dataAugmenter.keepOriginal else 0)
            repeatedTrainGlobalFeatures = self.trainGlobalFeaturesTensor[
                train_idx
            ].repeat(nCopies, 1)
            repeatedTrainLabels = self.trainLabelsTensor[train_idx].repeat(nCopies)

            # Create labelled training dataset
            trainDataset = PirateDataset(
                timeSeriesTensor=augmentedTrainTimeSeries,
                globalFeaturesTensor=repeatedTrainGlobalFeatures,
                labelsTensor=repeatedTrainLabels,
            )

            # Apply data augmentation to validation data
            augmentedValTimeSeries = dataAugmenter.augment(
                self.trainTimeSeriesTensor[val_idx], dim=1
            )
            augmentedValTimeSeries = windowingAugmenter.augment(
                augmentedValTimeSeries, dim=1
            )

            # Repeat global features and labels accordingly for validation data
            repeatedValGlobalFeatures = self.trainGlobalFeaturesTensor[val_idx].repeat(
                nCopies, 1
            )
            repeatedValLabels = self.trainLabelsTensor[val_idx].repeat(nCopies)

            # Create labelled validation dataset
            valDataset = PirateDataset(
                timeSeriesTensor=augmentedValTimeSeries,
                globalFeaturesTensor=repeatedValGlobalFeatures,
                labelsTensor=repeatedValLabels,
            )

            if includeTestInFolds:
                # Combine with unlabeled dataset for training
                trainDataset = ConcatDataset([trainDataset, self.unlabeledDataset])

            self.folds.append((trainDataset, valDataset))

        print(f"[PirateDataModule] K-Folds setup completed with {nFolds} folds.")

    def getFoldDataLoaders(self, foldIndex: int):
        if not hasattr(self, "folds"):
            raise ValueError("K-Folds not set up. Call setupKFolds() first.")

        if foldIndex < 0 or foldIndex >= len(self.folds):
            raise ValueError(
                f"Out of bounds foldIndex: got {foldIndex} for {len(self.folds)} folds."
            )

        trainSubset, valSubset = self.folds[foldIndex]

        trainLoader = TorchDataLoader(
            trainSubset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        valLoader = TorchDataLoader(
            valSubset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return trainLoader, valLoader

    def getTestLoader(self, windowAugmenter: WindowingAugmenter):
        # TODO: add option for TestTimeAugmentation
        # Apply windowing to test data
        augmentedTestTimeSeries = windowAugmenter.augment(
            self.testTimeSeriesTensor, dim=1
        )

        testDataset = PirateDataset(
            timeSeriesTensor=augmentedTestTimeSeries,
            globalFeaturesTensor=self.testGlobalFeaturesTensor,
            labelsTensor=-torch.ones(
                self.testTimeSeriesTensor.shape[0], dtype=torch.long
            ),  # dummy labels
        )

        testLoader = TorchDataLoader(
            testDataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return testLoader
