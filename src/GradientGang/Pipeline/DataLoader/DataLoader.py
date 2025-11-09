import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset, random_split
import pandas as pd
import os
import numpy as np
from ..Utils.ParameterInterpreter import ParameterInterpreter


class TimeSeriesAndGlobalDataset(Dataset):
    @staticmethod
    def fromCSV(
        dataPath: str,
        labelsPath: str | None = None,
        primaryKeyColumn: str = "sample_index",
        globalColumns: list[str] = [],
        timeSeriesColumns: list[str] | None = None,
        labelMapping: list[str] = ["no_pain", "low_pain", "high_pain"],
    ) -> "TimeSeriesAndGlobalDataset":
        # Load data from CSV
        data_df = pd.read_csv(dataPath)

        globalColumns = data_df.columns.intersection(globalColumns).tolist()

        if "time" in data_df.columns:
            globalFeatures = (
                data_df.groupby(primaryKeyColumn).first()[globalColumns].to_numpy()
            )
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)

            if timeSeriesColumns is None:
                timeSeriesColumns = data_df.columns.difference(
                    [primaryKeyColumn, "time"] + globalColumns
                ).tolist()

            timeSeriesDF = data_df[[primaryKeyColumn, "time"] + timeSeriesColumns]
            timeSeries = np.stack(
                [
                    timeSeriesDF.pivot(
                        index=primaryKeyColumn, columns="time", values=feat
                    ).to_numpy()
                    for feat in timeSeriesColumns
                ],
                axis=-1,
            )
            timeSeries = torch.tensor(timeSeries, dtype=torch.float32)
            timeSeries = timeSeries.permute(0, 2, 1)  # (samples, features, time)
        else:
            globalFeatures = data_df[globalColumns].to_numpy()
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
            timeSeries = None

        if labelsPath is not None:
            labels_df = pd.read_csv(labelsPath)
            # Map string labels to integers if necessary
            labels_df["label"] = labels_df["label"].map(
                {label: idx for idx, label in enumerate(labelMapping)}
            )
            labels = torch.tensor(labels_df["label"].to_numpy(), dtype=torch.long)

            # one hot encoding
            labelsOneHot = torch.zeros(
                (labels.shape[0], len(labelMapping)), dtype=torch.float32
            )
            labelsOneHot.scatter_(1, labels.unsqueeze(1), 1.0)

            labels = labelsOneHot
        else:
            labels = None

        return TimeSeriesAndGlobalDataset(timeSeries, globalFeatures, labels)

    def __init__(
        self,
        time_series_data: torch.Tensor | None,
        global_data: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        if time_series_data is not None:
            assert len(time_series_data) == len(
                global_data
            ), "All inputs must have the same number of samples."

        if labels is not None:
            assert len(global_data) == len(
                labels
            ), "All inputs must have the same number of samples."

        self.time_series_data = time_series_data
        self.global_data = global_data
        self.labels = labels

    def __len__(self):
        return self.global_data.shape[0]

    def __getitem__(self, index):
        return (
            self.time_series_data[index] if self.time_series_data is not None else None,
            self.global_data[index],
        ), (self.labels[index] if self.labels is not None else None)


class DataModule(L.LightningDataModule):
    """
    DataModule for loading datasets from CSV files and preparing PyTorch DataLoaders.
    Supports training, validation, and testing splits, as well as label mapping.
    """

    DataLoaderInterpreter = ParameterInterpreter(
        name="DataLoaderInterpreter",
        interpretation={},
        requiredParams={
            "data_dir": str,
            "train_file_name": str,
            "train_file_name_labels": str,
            "test_file_name": str,
            "batch_size": int,
            "num_workers": int,
            "data_dir": str,
            "train_file_name": str,
            "train_file_name_labels": str,
            "test_file_name": str,
            "batch_size": int,
            "num_workers": int,
            "val_split": float,
        },
    )

    def __init__(self, params: dict):
        super().__init__()
        self.DataLoaderInterpreter.checkRequiredParams(params)

        self.data_dir = params.get("data_dir", "")
        self.train_file_name = params.get("train_file_name", "")
        self.train_file_name_labels = params.get("train_file_name_labels", "")
        self.test_file_name = params.get("test_file_name", "")
        self.batch_size = params.get("batch_size", 32)
        self.num_workers = params.get("num_workers", 0)
        self.val_split = params.get("val_split", 0.1)

        self.globalFeaturesColumns = params.get(
            "globalFeaturesColumns", ["isPirate", "isNotPirate"]
        )
        self.primaryKeyColumn = params.get("primaryKeyColumn", "sample_index")
        self.timeSeriesColumns = params.get("timeSeriesColumns", None)

        # Label mapping for converting string labels to integers
        # Default mapping for pirate pain dataset
        self.label_mapping = params.get(
            "label_mapping", {"no_pain": 0, "low_pain": 1, "high_pain": 2}
        )

    def setup(self, stage: str | None = None):
        """
        Setup datasets for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            full_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.train_file_name),
                labelsPath=os.path.join(self.data_dir, self.train_file_name_labels),
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
            )

            # Split into train and validation
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [
                    train_size,
                    val_size,
                ],
                generator=torch.Generator().manual_seed(42),  # For reproducibility
            )

        if stage == "test" or stage is None:
            self.test_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.test_file_name),
                labelsPath=None,
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
            )

    def train_dataloader(self):
        """
        Create training data loader.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """
        Create validation data loader.
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        """
        Create test data loader.
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        """
        Create prediction data loader (uses test dataset).
        """
        return self.test_dataloader()
