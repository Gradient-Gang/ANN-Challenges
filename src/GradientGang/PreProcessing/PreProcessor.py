import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm
import seaborn as sns

sns.set_theme()


class PreProcessor:
    @staticmethod
    def fromYAML(path: str):
        """Builds a PreProcessor from a YAML file

        Args:
            path (str): The path of the YAML file

        Returns:
            PreProcessor: The built PreProcessor
        """
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return PreProcessor(params)

    def __init__(self, params: dict):
        """
        Initialize the PreProcessing class by loading parameters from a YAML file.
        """
        self.params = params

        # paths
        self.path_raw_data = self.params.get("path_raw_data", "")
        self.path_processed_data = self.params.get("path_processed_data", "")

        # file names
        self.name_train_file = self.params.get("name_train_file", "train.csv")
        self.name_test_file = self.params.get("name_test_file", "test.csv")
        self.name_train_labels_file = self.params.get(
            "name_train_labels_file", "train_labels.csv"
        )

        # feature handling flags
        self.drop_all_is_pirate = self.params.get("drop_all_is_pirate", False)
        self.one_hot_encode_is_pirate = self.params.get("one_hot_encode", False)

        # PCA
        self.use_pca = self.params.get("PCA", False)
        if self.use_pca:
            self.explained_variance = self.params.get("explained_variance", 0.95)

        # feature selection
        self.use_feature_selection = self.params.get("feature_selection", False)
        if self.use_feature_selection:
            self.feature_selected = self.params.get("feature_selected", None)

        # Columns not to be normalized
        self.columns_excluded_from_normalization: list[str] = self.params.get(
            "columns_excluded_from_normalization",
            ["sample_index", "time", "isPirate", "isNotPirate"],
        )

        # verbosity
        self.verbose = self.params.get("verbose", True)

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """
        file_path = os.path.join(self.path_raw_data, file_name)
        data = pd.read_csv(file_path)
        return data

    def save_data(self, data, file_name: str):
        """
        Save data to a CSV file. Accepts DataFrame or NumPy array.
        """
        os.makedirs(self.path_processed_data, exist_ok=True)

        file_path = os.path.join(self.path_processed_data, file_name)
        if not isinstance(data, pd.DataFrame):
            # convert NumPy array or other array-like to DataFrame
            data = pd.DataFrame(data)

        data.to_csv(file_path, index=False)

    def remove_last_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the last column from the DataFrame.
        """
        return data.drop(data.columns[-1], axis=1)

    def handle_is_pirate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle isPirate features by dropping or encoding them.
        """
        # If user wants to drop everything related to 'is_pirate' features
        if self.drop_all_is_pirate:
            return data.drop(columns=["n_legs", "n_hands", "n_eyes"], errors="ignore")

        data = data.drop(columns=["n_legs", "n_hands"], errors="ignore")

        # default: map n_eyes to a numeric column called 'number' then drop originals
        eye_map = {"two": 0, "one+eye_patch": 1}

        data["isPirate"] = data["n_eyes"].map(eye_map).fillna(0).astype(int)

        data = data.drop(columns=["n_legs", "n_hands", "n_eyes"], errors="ignore")

        if self.one_hot_encode_is_pirate:
            data["isNotPirate"] = 1 - data["isPirate"]
        return data

    def normalize_per_process(
        self, training_data: pd.DataFrame, test_data: pd.DataFrame
    ):
        """
        Normalize numerical features to have zero mean and unit variance.
        Returns (training_data, test_data).
        """
        # Select numeric columns (includes integers and floats)
        columns = training_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude specified columns from normalization
        for col in self.columns_excluded_from_normalization:
            if col in columns:
                columns.remove(col)

        # Normalize each column with safe handling for zero std
        for col in columns:
            mean = training_data[col].mean()
            std = training_data[col].std()
            if std == 0 or np.isnan(std):
                # center only; keep zeros for constant feature in train, center test
                training_data[col] = training_data[col] - mean
                test_data[col] = test_data[col] - mean
            else:
                training_data[col] = (training_data[col] - mean) / std
                test_data[col] = (test_data[col] - mean) / std

        return training_data, test_data

    def plot_one_time_series(self, data: pd.DataFrame, number: int):
        """
        Plot a single time series from the DataFrame.
        """
        Set = [i for i in range(number)]
        T = [data[data["sample_index"] == i] for i in Set]

        print(f"Selected sample indices: {Set}")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for ti in T:
            axes[0].plot(ti["time"], ti["pain_survey_1"])
            axes[1].plot(ti["time"], ti["pain_survey_2"])
            axes[2].plot(ti["time"], ti["pain_survey_3"])
            axes[3].plot(ti["time"], ti["pain_survey_4"])
            axes[4].plot(ti["time"], ti["joint_00"])
            axes[5].plot(ti["time"], ti["joint_01"])
            axes[6].plot(ti["time"], ti["joint_28"])
            axes[7].plot(ti["time"], ti["joint_29"])

        axes[0].set(title="pain survey 1", xlabel="time", ylabel="pain_survey_1")
        axes[1].set(title="pain survey 2", xlabel="time", ylabel="pain_survey_2")
        axes[2].set(title="pain survey 3", xlabel="time", ylabel="pain_survey_3")
        axes[3].set(title="pain survey 4", xlabel="time", ylabel="pain_survey_4")
        axes[4].set(title="joint 00", xlabel="time", ylabel="joint_00")
        axes[5].set(title="joint 01", xlabel="time", ylabel="joint_01")
        axes[6].set(title="joint 28", xlabel="time", ylabel="joint_28")
        axes[7].set(title="joint 29", xlabel="time", ylabel="joint_29")

        plt.tight_layout()
        plt.show()

    def __aggregate_time_series(
        self,
        data: pd.DataFrame,
        primaryKeyColumn: str,
        timeColumn: str,
        extraColumns: list[str],
    ) -> np.ndarray:
        data = data[data.columns.difference(extraColumns)]

        npData = np.stack(
            [
                data.pivot(index="sample_index", columns="time", values=feat).to_numpy()
                for feat in data.columns.difference([primaryKeyColumn, timeColumn])
            ],
            axis=-1,
        )

        return npData

    def __plotPcaNumComponentsPerFeature(
        self, pcadata: list[np.ndarray], feature_names: list[str]
    ):
        """
        Plot the number of PCA components selected per feature.
        """
        num_components = [data.shape[1] for data in pcadata]

        plt.figure(figsize=(10, 6))
        plt.bar(
            feature_names,
            num_components,
            color=sns.color_palette("viridis", len(feature_names)),
            align="edge",
        )
        plt.xlabel("Feature")
        plt.ylabel("Number of PCA Components")
        plt.title("PCA Components per Feature")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def apply_pca(self, training_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Apply PCA to reduce dimensionality of the data.
        Returns (training_data, test_data).
        """

        trainingDataNp = self.__aggregate_time_series(
            training_data, "sample_index", "time", ["isPirate", "isNotPirate"]
        )

        testDataNp = self.__aggregate_time_series(
            test_data, "sample_index", "time", ["isPirate", "isNotPirate"]
        )

        maxTrainingIndex = trainingDataNp.shape[0]

        dataNp = np.concatenate((trainingDataNp, testDataNp), axis=0)

        pcaData = []

        for i in tqdm(range(dataNp.shape[2]), desc="Applying PCA"):
            # Apply PCA on each feature separately
            featureSlice = dataNp[:, :, i]

            pca = PCA(n_components=self.explained_variance)
            featureSliceTransformed = pca.fit_transform(featureSlice)

            pcaData.append(featureSliceTransformed)

        if self.verbose:
            feature_names = training_data.columns.difference(
                ["sample_index", "time", "isPirate", "isNotPirate"]
            ).tolist()
            self.__plotPcaNumComponentsPerFeature(pcaData, feature_names)

        pcaData = np.concatenate(pcaData, axis=1)

        trainingPcaData = pcaData[:maxTrainingIndex]

        training_data.drop(
            columns=training_data.columns.difference(
                [
                    "sample_index",
                ]
                + ["isPirate", "isNotPirate"]
            ),
            inplace=True,
        )
        training_data = training_data.groupby("sample_index").first().reset_index()
        training_data_pca = pd.DataFrame(trainingPcaData)

        training_data = pd.concat(
            [training_data.reset_index(drop=True), training_data_pca], axis=1
        )

        testPcaData = pcaData[maxTrainingIndex:]
        test_data.drop(
            columns=test_data.columns.difference(
                [
                    "sample_index",
                ]
                + ["isPirate", "isNotPirate"]
            ),
            inplace=True,
        )
        test_data = test_data.groupby("sample_index").first().reset_index()
        test_data_pca = pd.DataFrame(testPcaData)
        test_data = pd.concat([test_data.reset_index(drop=True), test_data_pca], axis=1)

        return training_data, test_data

    def apply_feature_selection(
        self, training_data: pd.DataFrame, test_data: pd.DataFrame
    ):
        """
        Select specific features from the data.
        """
        training_data = training_data[self.feature_selected]
        test_data = test_data[self.feature_selected]
        return training_data, test_data

    def preprocess(self):
        """
        Main preprocessing function to load, process, and save data.
        """
        try:
            train_data = self.load_data(self.name_train_file)
            test_data = self.load_data(self.name_test_file)
            train_labels = self.load_data(self.name_train_labels_file)
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        print("Data loaded successfully.")

        try:
            train_data = self.remove_last_column(train_data)
            test_data = self.remove_last_column(test_data)
        except Exception as e:
            print(f"Error removing last column: {e}")
            return
        print("Last column removed successfully.")

        try:
            train_data = self.handle_is_pirate_features(train_data)
            test_data = self.handle_is_pirate_features(test_data)
        except Exception as e:
            print(f"Error handling inspirate features: {e}")
            return
        print("Inspirate features handled successfully.")

        if self.verbose:
            try:
                self.plot_one_time_series(train_data, number=3)
            except Exception:
                # plotting should not stop preprocessing
                pass

        try:
            train_data, test_data = self.normalize_per_process(train_data, test_data)
        except Exception as e:
            print(f"Error normalizing data: {e}")
            return
        print("Data normalized successfully.")

        if self.verbose:
            try:
                self.plot_one_time_series(train_data, number=3)
            except Exception:
                pass

        if self.use_pca:
            try:
                train_data, test_data = self.apply_pca(train_data, test_data)
            except Exception as e:
                print(f"Error applying PCA: {e}")
                return
            print("PCA applied successfully.")
        else:
            print("PCA not applied.")

        if self.use_feature_selection:
            try:
                train_data, test_data = self.apply_feature_selection(
                    train_data, test_data
                )
            except Exception as e:
                print(f"Error applying feature selection: {e}")
                return
            print("Feature selection applied successfully.")
        else:
            print("Feature selection not applied.")

        try:
            self.save_data(train_data, self.name_train_file)
            self.save_data(test_data, self.name_test_file)
            self.save_data(train_labels, self.name_train_labels_file)
        except Exception as e:
            print(f"Error saving data: {e}")
        print("Data saved successfully.")
