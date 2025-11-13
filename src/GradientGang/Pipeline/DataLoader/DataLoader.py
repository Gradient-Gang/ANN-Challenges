import pytorch_lightning as L
import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    random_split,
    Subset,
)
import pandas as pd
import os
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from sklearn.model_selection import KFold
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
        globalFeaturesPath: str | None = None,
    ) -> "TimeSeriesAndGlobalDataset":
        """
        Create a TimeSeriesAndGlobalDataset from CSV files.
        Args:
            dataPath (str): Path to the CSV file containing the data.
            labelsPath (str | None): Path to the CSV file containing the labels. If None, no labels are loaded.
            primaryKeyColumn (str): Column name for the primary key.
            globalColumns (list[str]): List of column names for global features.
            timeSeriesColumns (list[str] | None): List of column names for time series features
            labelMapping (list[str]): List of possible labels for mapping string labels to integers.
            globalFeaturesPath (str | None): Path to CSV file containing pre-extracted global features (e.g., from PreProcessor)
        Returns:
            TimeSeriesAndGlobalDataset: The constructed dataset.
        """

        # Load data from CSV and check for required columns
        data_df = pd.read_csv(dataPath)

        # Flag to track if global features were loaded from separate file
        global_features_loaded_separately = False

        # If separate global features file is provided, load and merge it
        if globalFeaturesPath is not None:
            try:
                global_features_df = pd.read_csv(globalFeaturesPath)
                # Extract sample indices in order from time series data
                sample_ids = data_df[primaryKeyColumn].unique()
                # Reindex global features to match time series sample order
                global_features_df = (
                    global_features_df.set_index(primaryKeyColumn)
                    .reindex(sample_ids)
                    .reset_index()
                )
                # All columns except sample_index are global features
                globalColumns = [
                    col for col in global_features_df.columns if col != primaryKeyColumn
                ]
                globalFeatures = global_features_df[globalColumns].to_numpy()
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
                global_features_loaded_separately = True
            except Exception as e:
                print(
                    f"Warning: Could not load global features from {globalFeaturesPath}: {e}"
                )
                print(
                    "Falling back to extracting global features from time series data."
                )
                globalColumns = data_df.columns.intersection(globalColumns).tolist()
                globalFeatures = (
                    data_df.groupby(primaryKeyColumn).first()[globalColumns].to_numpy()
                )
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
        else:
            # Extract global features from time series data (original behavior)
            globalColumns = data_df.columns.intersection(globalColumns).tolist()
            globalFeatures = (
                data_df.groupby(primaryKeyColumn).first()[globalColumns].to_numpy()
            )
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)

        if timeSeriesColumns is None:
            timeSeriesColumns = data_df.columns.difference(
                [primaryKeyColumn, "time"] + globalColumns
            ).tolist()

        # Build an ordered list of sample primary keys and features
        if "time" in data_df.columns:
            # Use groupby first to obtain one row per sample and capture the sample order
            grouped = data_df.groupby(primaryKeyColumn).first()
            sample_ids = grouped.index.to_numpy()

            # Only extract global features from time series if they weren't loaded separately
            if not global_features_loaded_separately:
                globalFeatures = grouped[globalColumns].to_numpy()
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)

            # Process time series data
            if timeSeriesColumns is None:
                timeSeriesColumns = data_df.columns.difference(
                    [primaryKeyColumn, "time"] + globalColumns
                ).tolist()

            # Pivot and stack time series data
            timeSeriesDF = data_df[[primaryKeyColumn, "time"] + timeSeriesColumns]
            # Pivot each feature and reindex to ensure the same sample order as grouped
            timeSeries_list = []
            for feat in timeSeriesColumns:
                pivoted = timeSeriesDF.pivot(
                    index=primaryKeyColumn, columns="time", values=feat
                ).reindex(sample_ids)
                timeSeries_list.append(pivoted.to_numpy())

            timeSeries = np.stack(timeSeries_list, axis=-1)
            timeSeries = torch.tensor(timeSeries, dtype=torch.float32)
            timeSeries = timeSeries.permute(0, 2, 1)  # (samples, features, time)

        # No time series data
        else:
            # No time-series dimension; preserve row order
            if primaryKeyColumn in data_df.columns:
                sample_ids = data_df[primaryKeyColumn].to_numpy()
            else:
                sample_ids = np.arange(len(data_df))

            globalFeatures = data_df[globalColumns].to_numpy()
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
            timeSeries = None

        # Load and align labels to the sample order (use -1 for unlabeled)
        if labelsPath is not None:
            labels_df = pd.read_csv(labelsPath)
            # Map textual labels to integers
            label_map = {label: idx for idx, label in enumerate(labelMapping)}

            if primaryKeyColumn in labels_df.columns:
                labels_series = labels_df.set_index(primaryKeyColumn)["label"].map(
                    label_map
                )
            else:
                # If no primary key in labels file, assume same order as samples
                labels_series = labels_df["label"].map(label_map)

            # Align labels to sample_ids and fill missing with -1
            labels_aligned = (
                pd.Series(sample_ids).map(labels_series).fillna(-1).astype(int)
            )
            labels = torch.tensor(labels_aligned.to_numpy(), dtype=torch.long)
        else:
            labels = torch.full((len(sample_ids),), -1, dtype=torch.long)

        # Return the constructed dataset
        return TimeSeriesAndGlobalDataset(timeSeries, globalFeatures, labels)

    def __init__(
        self,
        time_series_data: torch.Tensor | None,
        global_data: torch.Tensor,
        labels: torch.Tensor | None,
    ):
        """
        Initialize the TimeSeriesAndGlobalDataset.
        Args:
            time_series_data (torch.Tensor | None): Time series data tensor or None if not available.
            global_data (torch.Tensor): Global features tensor.
            labels (torch.Tensor | None): Labels tensor or None if not available.
        """

        # Validate input dimensions
        if time_series_data is not None:
            assert len(time_series_data) == len(
                global_data
            ), "All inputs must have the same number of samples."

        # Validate labels length
        if labels is not None:
            assert len(global_data) == len(
                labels
            ), "All inputs must have the same number of samples."

        # Store data
        self.time_series_data = time_series_data
        self.global_data = global_data
        self.labels = labels

        # Store dataset Information for quick access
        self.timeSeriesShape = (
            time_series_data.shape[1:] if time_series_data is not None else None
        )
        self.globalFeaturesShape = global_data.shape[1:]
        self.numClasses = (
            (labels.max().item() + 1) if labels is not None and len(labels) > 0 else 0
        )

    def __len__(self):
        # Return the number of samples in the dataset
        return self.global_data.shape[0]

    def __getitem__(self, index):
        # Retrieve the sample at the specified index
        return (
            self.time_series_data[index] if self.time_series_data is not None else None,
            self.global_data[index],
        ), (self.labels[index] if self.labels is not None else None)

    def __add__(self, other: Dataset) -> ConcatDataset:
        return ConcatDataset([self, other])


class DataModule(L.LightningDataModule):
    """
    DataModule for loading datasets from CSV files and preparing PyTorch DataLoaders.
    Supports training, validation, and testing splits, as well as label mapping.
    """

    # Parameter interpreter for validating input parameters
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
            "val_split": float,
        },
    )

    def __init__(self, params: dict):
        """
        Initialize the DataModule with parameters.
        Args:
            params (dict): Dictionary of parameters for DataModule configuration.
        """

        super().__init__()
        self.DataLoaderInterpreter.checkRequiredParams(params)

        # Initialize attributes from parameters
        self.data_dir = params.get("data_dir", "")
        self.train_file_name = params.get("train_file_name", "")
        self.train_file_name_labels = params.get("train_file_name_labels", "")
        self.test_file_name = params.get("test_file_name", "")
        self.batch_size = params.get("batch_size", 32)
        self.num_workers = params.get("num_workers", 0)
        self.val_split = params.get("val_split", 0.1)

        # Initialize feature columns and primary key
        self.globalFeaturesColumns = params.get(
            "globalFeaturesColumns", ["isPirate", "isNotPirate"]
        )
        self.primaryKeyColumn = params.get("primaryKeyColumn", "sample_index")
        self.timeSeriesColumns = params.get("timeSeriesColumns", None)

        # Optional: Separate file for global features
        self.train_global_features_file = params.get("train_global_features_file", None)
        self.test_global_features_file = params.get("test_global_features_file", None)

        # Label mapping for converting string labels to integers
        # Default mapping for pirate pain dataset
        self.label_mapping = params.get(
            "label_mapping", {"no_pain": 0, "low_pain": 1, "high_pain": 2}
        )

        # Random seed for train/val split (can be overridden via params)
        split_seed = params.get("split_seed", 42)
        self.trainValGenerator = torch.Generator().manual_seed(split_seed)

        # K-Fold Cross-Validation settings
        self.use_kfold = params.get("use_kfold", False)
        self.n_folds = params.get("n_folds", 5)
        self.current_fold = None
        self._full_labeled_dataset = None  # Store full dataset for K-Fold splitting

    def setup(self, stage: str | None = None, includeTestInTrain: bool = True):
        """
        Setup datasets for training, validation, and testing.
        Args:
            stage (str | None): Stage of setup ("fit", "test", or None for all).
        """

        # Load and split datasets based on the stage
        # if stage is "fit", load training and validation datasets
        if stage == "fit" or stage is None:
            # Determine global features file path
            train_global_path = None
            if self.train_global_features_file:
                train_global_path = os.path.join(
                    self.data_dir, self.train_global_features_file
                )

            # Load labeled training dataset and unlabeled test dataset separately
            labeled_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.train_file_name),
                labelsPath=os.path.join(self.data_dir, self.train_file_name_labels),
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
                globalFeaturesPath=train_global_path,
            )

            self.updateDataInfo(labeled_dataset)

            # Store full labeled dataset for K-Fold splitting
            self._full_labeled_dataset = labeled_dataset

            # Split labeled dataset into train/val (do not mix unlabeled test into this split)
            # If using K-Fold, this will be overridden by setup_fold()
            val_size = int(len(labeled_dataset) * self.val_split)
            train_size = len(labeled_dataset) - val_size

            self.train_labeled, self.val_dataset = random_split(
                labeled_dataset,
                [train_size, val_size],
                generator=self.trainValGenerator,
            )

            if includeTestInTrain:
                # Determine test global features file path
                test_global_path = None
                if self.test_global_features_file:
                    test_global_path = os.path.join(
                        self.data_dir, self.test_global_features_file
                    )

                unlabeled_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                    dataPath=os.path.join(self.data_dir, self.test_file_name),
                    labelsPath=None,
                    labelMapping=list(self.label_mapping.keys()),
                    globalColumns=self.globalFeaturesColumns,
                    primaryKeyColumn=self.primaryKeyColumn,
                    timeSeriesColumns=self.timeSeriesColumns,
                    globalFeaturesPath=test_global_path,
                )

                # For reconstruction training we allow unlabeled test data to be mixed with labeled train data
                self.train_dataset = ConcatDataset(
                    [self.train_labeled, unlabeled_dataset]
                )
            else:
                self.train_dataset = self.train_labeled

        if stage == "test" or stage is None:
            # Determine test global features file path
            test_global_path = None
            if self.test_global_features_file:
                test_global_path = os.path.join(
                    self.data_dir, self.test_global_features_file
                )

            self.test_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.test_file_name),
                labelsPath=None,
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
                globalFeaturesPath=test_global_path,
            )

    def updateDataInfo(self, dataset: TimeSeriesAndGlobalDataset) -> dict:
        self.dataInfo = {
            "timeSeriesShape": dataset.timeSeriesShape,
            "globalFeaturesShape": dataset.globalFeaturesShape,
            "numClasses": dataset.numClasses,
        }

    def train_dataloader(self):
        """
        Create training data loader.
        """

        # Check if training dataset is initialized
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        # Return the DataLoader for training dataset
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

        # Check if validation dataset is initialized
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        # Return the DataLoader for validation dataset
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

        # Check if test dataset is initialized
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        # Return the DataLoader for test dataset
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
        # Return the DataLoader for prediction dataset (which is the test dataset)
        return self.test_dataloader()

    def getDatasetInfo(self) -> dict:
        """
        Get information about the dataset, including input shapes and number of classes.
        Returns:
            dict: Dictionary containing dataset information.
        """

        return self.dataInfo

    def setup_fold(self, fold_idx: int, include_test_in_train: bool = True):
        """
        Setup train/val datasets for a specific K-Fold split.

        Args:
            fold_idx (int): Index of the fold to setup (0 to n_folds-1).
            include_test_in_train (bool): Whether to include unlabeled test data in training set.

        Note:
            Must call setup(stage='fit') before using this method to load the full dataset.
        """
        if not self.use_kfold:
            raise RuntimeError("K-Fold is not enabled. Set use_kfold=True in params.")

        if self._full_labeled_dataset is None:
            raise RuntimeError(
                "Full labeled dataset not loaded. Call setup(stage='fit') first."
            )

        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise ValueError(
                f"fold_idx must be between 0 and {self.n_folds-1}, got {fold_idx}"
            )

        # Create K-Fold splitter
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Get train/val indices for this fold
        all_indices = list(range(len(self._full_labeled_dataset)))
        splits = list(kfold.split(all_indices))
        train_indices, val_indices = splits[fold_idx]

        # Create train and val subsets
        self.train_labeled = Subset(self._full_labeled_dataset, train_indices)
        self.val_dataset = Subset(self._full_labeled_dataset, val_indices)

        # Optionally include unlabeled test data in training
        if (
            include_test_in_train
            and hasattr(self, "test_dataset")
            and self.test_dataset is not None
        ):
            self.train_dataset = ConcatDataset([self.train_labeled, self.test_dataset])
        else:
            self.train_dataset = self.train_labeled

        # Update current fold tracker
        self.current_fold = fold_idx

    def get_fold_info(self) -> dict:
        """
        Get information about the current K-Fold setup.

        Returns:
            dict: Dictionary containing K-Fold configuration and current fold index.
        """
        return {
            "use_kfold": self.use_kfold,
            "n_folds": self.n_folds,
            "current_fold": self.current_fold,
            "full_dataset_size": (
                len(self._full_labeled_dataset) if self._full_labeled_dataset else None
            ),
            "train_size": (
                len(self.train_labeled) if hasattr(self, "train_labeled") else None
            ),
            "val_size": len(self.val_dataset) if hasattr(self, "val_dataset") else None,
        }
