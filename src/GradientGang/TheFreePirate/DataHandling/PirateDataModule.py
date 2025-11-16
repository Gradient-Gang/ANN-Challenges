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
        # Apply data augmentation to training data
        augmentedTimeSeries = dataAugmenter.augment(self.trainTimeSeriesTensor, dim=1)
        augmentedTimeSeries = windowingAugmenter.augment(augmentedTimeSeries, dim=1)

        # Repeat global features and labels accordingly
        nCopies = dataAugmenter.nCopies + (1 if dataAugmenter.keepOriginal else 0)
        repeatedGlobalFeatures = self.trainGlobalFeaturesTensor.repeat(nCopies, 1)
        repeatedLabels = self.trainLabelsTensor.repeat(nCopies)

        print("[PirateDataModule] Data augmentation completed ")
        # Create labelled dataset
        self.labeledDataset = PirateDataset(
            timeSeriesTensor=augmentedTimeSeries,
            globalFeaturesTensor=repeatedGlobalFeatures,
            labelsTensor=repeatedLabels,
        )

        # Create unlabeled dataset from test data
        if augmentTestSet:
            augmentedTestTimeSeries = dataAugmenter.augment(
                self.testTimeSeriesTensor, dim=1
            )
        else:
            augmentedTestTimeSeries = self.testTimeSeriesTensor

        augmentedTestTimeSeries = windowingAugmenter.augment(
            augmentedTestTimeSeries, dim=1
        )

        repeatedGlobalFeatures = self.testGlobalFeaturesTensor.repeat(nCopies, 1)

        self.unlabeledDataset = PirateDataset(
            timeSeriesTensor=augmentedTestTimeSeries,
            globalFeaturesTensor=repeatedGlobalFeatures,
            labelsTensor=-torch.ones(
                self.testTimeSeriesTensor.shape[0] * nCopies, dtype=torch.long
            ),  # dummy labels
        )

        # Prepare stratified K-Fold splits
        skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)
        self.folds = []

        X_indices = np.arange(len(self.labeledDataset))
        y_labels = repeatedLabels.numpy()

        for train_idx, val_idx in skf.split(X_indices, y_labels):
            trainSubset = Subset(self.labeledDataset, train_idx)
            if includeTestInFolds:
                # Combine with unlabeled dataset
                trainSubset = ConcatDataset([trainSubset, self.unlabeledDataset])

            valSubset = Subset(self.labeledDataset, val_idx)
            self.folds.append((trainSubset, valSubset))

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
