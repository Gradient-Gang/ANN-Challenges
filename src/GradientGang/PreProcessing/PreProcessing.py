import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sklearn

class PreProcessing:
    def __init__(self, path_params: str):
        """
        Initialize the PreProcessing class by loading parameters from a YAML file.
        """
        with open(path_params, 'r') as f:
            self.params = yaml.safe_load(f)

        #print(self.params)

        self.path_raw_data = self.params.get("path_raw_data", "")
        self.path_processed_data = self.params.get("path_processed_data", "")

        self.name_train_file = self.params.get("name_train_file", "train.csv")
        self.name_test_file = self.params.get("name_test_file", "test.csv")
        self.name_train_labels_file = self.params.get("name_train_labels_file", "train_labels.csv")

        self.drop_all_is_pirate = self.params.get("drop_all_is_pirate", False)
        self.one_hot_encode_is_pirate = self.params.get("one_hot_encode", False)

        self.pca = self.params.get("pca", False)
        if self.pca:
            self.explained_variance = self.params.get("explained_variance", 0.95)

        self.feature_selection = self.params.get("feature_selection", False)
        if self.feature_selection:
            self.feature_selected = self.params.get("feature_selected", None)

        self.verbose = self.params.get("verbose", True)

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """
        file_path = f"{self.path_raw_data}/{file_name}"
        data = pd.read_csv(file_path)
        return data
    
    def save_data(self, data: pd.DataFrame, file_name: str):
        """
        Save data to a CSV file.
        """
        file_path = f"{self.path_processed_data}/{file_name}"
        data.to_csv(file_path, index=False)

    def remove_last_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the last column from the DataFrame.
        """
        return data.drop(data.columns[-1], axis=1)
    
    def handle_inspirate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle inspirate features by dropping or encoding them."""
        if self.drop_all_is_pirate:
            return data.drop(columns=["n_legs", "n_hands", "n_eyes"])
        
        
        eye_map = {"two": 2, "one+eye_patch": 1}
        data["number"] = data["n_eyes"].map(eye_map).fillna(0).astype(int)
        data = data.drop(columns=["n_legs", "n_hands", "n_eyes"])

        if self.one_hot_encode_is_pirate:
            data = pd.get_dummies(data, columns=["n_eyes"], drop_first=False)
            return data
        else:
            return data

    def normalize_per_process(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features to have zero mean and unit variance.
        """
        # Extract numerical columns:
        columns =[col for col in training_data.columns if training_data[col].dtype in [np.float64, np.float32]]

        # Normalize each column
        for col in columns:
            mean = training_data[col].mean()
            std = training_data[col].std()
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

        for i, ti in enumerate(T):
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
                
    def pca(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA to reduce dimensionality of the data.
        """

        pca = sklearn.decomposition.PCA(n_components=self.explained_variance)
        training_data = pca.fit_transform(training_data)
        test_data = pca.transform(test_data)

        return training_data, test_data

    def feature_selection(self, training_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Select specific features from the data.
        """
        training_data, test_data = training_data[self.feature_selected], test_data[self.feature_selected]
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
            train_data = self.handle_inspirate_features(train_data)
            test_data = self.handle_inspirate_features(test_data)
        except Exception as e:
            print(f"Error handling inspirate features: {e}")
            return
        print("Inspirate features handled successfully.")

        if self.verbose:
            self.plot_one_time_series(train_data, number=3)

        try:
            train_data, test_data = self.normalize_per_process(train_data, test_data)
        except Exception as e:
            print(f"Error normalizing data: {e}")
            return
        print("Data normalized successfully.")

        if self.verbose:
            self.plot_one_time_series(train_data, number=3)

        if self.pca:
            try:
                train_data, test_data = self.pca(train_data, test_data)
            except Exception as e:
                print(f"Error applying PCA: {e}")
                return
            print("PCA applied successfully.")
        else:
            print("PCA not applied.")

        if self.feature_selection:
            try:
                train_data, test_data = self.feature_selection(train_data, test_data)
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