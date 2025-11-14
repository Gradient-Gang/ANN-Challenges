import pytest
import optuna
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from GradientGang.Pipeline.FinalPipeline import FinalPipeline
from GradientGang.Pipeline.Architectures.Direct import Direct
from GradientGang.Pipeline.Architectures.LightningAutoencoder import LightningAutoencoder


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    lightning_logs = os.path.join(temp_dir, "lightning_logs")
    submission_dir = os.path.join(temp_dir, "submissions")
    
    os.makedirs(lightning_logs, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    yield {
        "temp_dir": temp_dir,
        "lightning_logs": lightning_logs,
        "submission_dir": submission_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_env_file(temp_dirs):
    """Create a mock .env file."""
    env_path = os.path.join(temp_dirs["temp_dir"], ".env")
    with open(env_path, "w") as f:
        f.write("DATABASE_URL=sqlite:///test.db\n")
    return env_path


@pytest.fixture
def basic_params(temp_dirs, mock_env_file):
    """Basic parameters for FinalPipeline initialization."""
    return {
        "project_name": "test_project",
        "study_name": "_test_study",
        "database_path": mock_env_file,
        "submission_path": temp_dirs["submission_dir"],
        "data_params": {
            "data_dir": "../dataset/PirateProcessed/",
            "train_file_name": "pirate_pain_train.csv",
            "train_file_name_labels": "pirate_pain_train_labels.csv",
            "test_file_name": "pirate_pain_test.csv",
            "train_global_features_file": "train_global_features.csv",
            "test_global_features_file": "test_global_features.csv",
            "batch_size": 32,
            "num_workers": 0,
            "val_split": 0.2,
            "shuffle": True,
            "use_kfold": True,
            "n_folds": 2,  # Small for testing
            "use_windowing": False,
            "window_size": 160,
            "stride": 160,
        }
    }


class TestFinalPipelineInitialization:
    """Test FinalPipeline initialization."""
    
    def test_initialization_success(self, basic_params):
        """Test successful initialization of FinalPipeline."""
        pipeline = FinalPipeline(basic_params)
        
        assert pipeline.project_name == "test_project"
        assert pipeline.study_name == "_test_study"
        assert pipeline.storage is not None
        assert pipeline.study is not None
        assert pipeline.best_model is None
        assert pipeline.dataloader is None
    
    def test_initialization_missing_project_name(self, basic_params):
        """Test initialization fails without project_name."""
        params = basic_params.copy()
        del params["project_name"]
        
        with pytest.raises(ValueError, match="Project name must be provided"):
            FinalPipeline(params)
    
    def test_initialization_missing_data_params(self, basic_params):
        """Test initialization fails without data_params."""
        params = basic_params.copy()
        del params["data_params"]
        
        with pytest.raises(ValueError, match="Data parameters must be provided"):
            FinalPipeline(params)
    
    def test_initialization_missing_submission_path(self, basic_params):
        """Test initialization fails without submission_path."""
        params = basic_params.copy()
        del params["submission_path"]
        
        with pytest.raises(ValueError, match="Submission path must be provided"):
            FinalPipeline(params)
    
    def test_initialization_missing_database_path(self, basic_params):
        """Test initialization fails without database_path."""
        params = basic_params.copy()
        del params["database_path"]
        
        with pytest.raises(ValueError, match="Database path must be provided"):
            FinalPipeline(params)


class TestStudyManagement:
    """Test study creation and management."""
    
    def test_create_study(self, basic_params):
        """Test study creation."""
        pipeline = FinalPipeline(basic_params)
        
        assert pipeline.study is not None
        assert pipeline.study.study_name == "test_project_test_study"
        assert pipeline.study.direction == optuna.study.StudyDirection.MAXIMIZE
    
    def test_study_summary_no_trials(self, basic_params, capsys):
        """Test study summary with no trials."""
        pipeline = FinalPipeline(basic_params)
        pipeline.study_summary()
        
        captured = capsys.readouterr()
        assert "Total trials: 0" in captured.out
        assert "No trials found" in captured.out


class TestLoadBestModel:
    """Test load_best_model functionality."""
    
    def test_load_best_model_no_trials(self, basic_params):
        """Test load_best_model fails when no trials exist."""
        pipeline = FinalPipeline(basic_params)
        
        with pytest.raises(ValueError, match="No completed trials found"):
            pipeline.load_best_model()
    
    @patch('GradientGang.Pipeline.FinalPipeline.DataModule')
    @patch('GradientGang.Pipeline.FinalPipeline.Direct')
    @patch('glob.glob')
    def test_load_best_model_with_mock_trial(
        self, mock_glob, mock_direct, mock_datamodule, basic_params, temp_dirs
    ):
        """Test load_best_model with a mocked completed trial."""
        pipeline = FinalPipeline(basic_params)
        
        # Create a mock completed trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.value = 0.85
        mock_trial.state = optuna.trial.TrialState.COMPLETE
        mock_trial.params = {
            'MacroArchitecture': 'Direct',
            'architectureType': 'Recurrent',
            'rnnType': 'LSTM',
            'hiddenDim': 64,
            'numLayers': 2,
            'bidirectional': False,
            'recurrentDropout': 0.2,
            'encoderActivation': 'ReLU',
            'globalEmbeddingDim': 32,
            'globalNumLayers': 1,
            'globalDropout': 0.1,
            'globalActivation': 'ReLU',
            'numFFLayers': 2,
            'ffHiddenDim': 64,
            'ffDropout': 0.2,
            'ffActivation': 'ReLU',
            'LearningRate': 0.001,
            'RegularizationWeight': 0.01,
            'SchedulerType': 'ReduceLROnPlateau',
            'SchedulerPatience': 5,
            'SchedulerFactor': 0.5,
            'SchedulerMinLR': 1e-6,
            'use_windowing': False,
        }
        
        # Mock DataModule
        mock_dm_instance = Mock()
        mock_dm_instance.setup = Mock()
        mock_dm_instance.getDatasetInfo.return_value = {
            'timeSeriesShape': (34, 160),
            'globalFeaturesShape': (32,),
            'numClasses': 3
        }
        mock_datamodule.return_value = mock_dm_instance
        
        # Mock checkpoint file
        checkpoint_path = os.path.join(
            temp_dirs["lightning_logs"], 
            "version_1", 
            "checkpoints", 
            "trial-1-fold-0-epoch=10-val_F1=0.850.ckpt"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        Path(checkpoint_path).touch()
        
        mock_glob.return_value = [checkpoint_path]
        
        # Mock Direct model
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_direct.load_from_checkpoint.return_value = mock_model_instance
        
        # Mock study.trials and best_trial using PropertyMock
        mock_trials_list = [mock_trial]
        with patch.object(pipeline.study, 'trials', mock_trials_list):
            with patch.object(pipeline.study, 'best_trial', mock_trial):
                # Load model
                model = pipeline.load_best_model()
                
                assert model is not None
                assert pipeline.best_model is not None
                assert pipeline.dataloader is not None
                mock_direct.load_from_checkpoint.assert_called_once()
    
    @patch('GradientGang.Pipeline.FinalPipeline.DataModule')
    @patch('glob.glob')
    def test_load_best_model_checkpoint_not_found(self, mock_glob, mock_datamodule, basic_params):
        """Test load_best_model fails when checkpoint file not found."""
        pipeline = FinalPipeline(basic_params)
        
        # Create a mock completed trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.value = 0.85
        mock_trial.state = optuna.trial.TrialState.COMPLETE
        mock_trial.params = {
            'MacroArchitecture': 'Direct',
            'architectureType': 'Recurrent',
            'rnnType': 'LSTM',
            'hiddenDim': 64,
            'numLayers': 2,
            'bidirectional': False,
            'recurrentDropout': 0.2,
            'encoderActivation': 'ReLU',
            'globalEmbeddingDim': 32,
            'globalNumLayers': 1,
            'globalDropout': 0.1,
            'globalActivation': 'ReLU',
            'numFFLayers': 2,
            'ffHiddenDim': 64,
            'ffDropout': 0.2,
            'ffActivation': 'ReLU',
            'LearningRate': 0.001,
            'RegularizationWeight': 0.01,
            'SchedulerType': 'ReduceLROnPlateau',
            'SchedulerPatience': 5,
            'SchedulerFactor': 0.5,
            'SchedulerMinLR': 1e-6,
            'use_windowing': False,
        }
        
        # Mock DataModule
        mock_dm_instance = Mock()
        mock_dm_instance.setup = Mock()
        mock_dm_instance.getDatasetInfo.return_value = {
            'timeSeriesShape': (34, 160),
            'globalFeaturesShape': (32,),
            'numClasses': 3
        }
        mock_datamodule.return_value = mock_dm_instance
        
        # Mock no checkpoints found
        mock_glob.return_value = []
        
        # Mock study.trials and best_trial
        mock_trials_list = [mock_trial]
        with patch.object(pipeline.study, 'trials', mock_trials_list):
            with patch.object(pipeline.study, 'best_trial', mock_trial):
                with pytest.raises(FileNotFoundError, match="No checkpoint found"):
                    pipeline.load_best_model()


class TestCreateSubmission:
    """Test create_submission functionality."""
    
    def test_create_submission_no_model_loads_best(self, basic_params):
        """Test create_submission automatically loads best model if not present."""
        pipeline = FinalPipeline(basic_params)
        
        # Mock load_best_model
        with patch.object(pipeline, 'load_best_model') as mock_load:
            mock_load.side_effect = ValueError("No completed trials")
            
            with pytest.raises(ValueError):
                pipeline.create_submission()
            
            mock_load.assert_called_once()
    
    @patch('GradientGang.Pipeline.FinalPipeline.WindowedSubmissionGenerator')
    def test_create_submission_with_model(self, mock_submitter_class, basic_params, temp_dirs):
        """Test create_submission with pre-loaded model."""
        pipeline = FinalPipeline(basic_params)
        
        # Mock best_model and dataloader
        mock_model = Mock()
        mock_dataloader = Mock()
        mock_test_loader = Mock()
        
        mock_dataloader.setup = Mock()
        mock_dataloader.test_dataloader.return_value = mock_test_loader
        
        pipeline.best_model = mock_model
        pipeline.dataloader = mock_dataloader
        
        # Mock submitter
        mock_submitter_instance = Mock()
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_submitter_instance.generate_submission.return_value = mock_df
        mock_submitter_class.return_value = mock_submitter_instance
        
        # Create submission
        result = pipeline.create_submission()
        
        assert result is not None
        mock_dataloader.setup.assert_called_once_with(stage='test', includeTestInTrain=False)
        mock_dataloader.test_dataloader.assert_called_once()
        mock_submitter_class.assert_called_once()
        mock_submitter_instance.generate_submission.assert_called_once()
    
    def test_create_submission_no_dataloader(self, basic_params):
        """Test create_submission fails if dataloader is None after model load."""
        pipeline = FinalPipeline(basic_params)
        
        # Set best_model but not dataloader (shouldn't happen in practice)
        pipeline.best_model = Mock()
        pipeline.dataloader = None
        
        with pytest.raises(ValueError, match="No dataloader found"):
            pipeline.create_submission()


class TestArchitectureSetup:
    """Test architecture setup methods."""
    
    def test_setUp_encoder_recurrent(self, basic_params):
        """Test setUpEncoder with Recurrent architecture."""
        pipeline = FinalPipeline(basic_params)
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            'architectureType': 'Recurrent',
            'rnnType': 'LSTM',
            'bidirectional': False,
            'encoderActivation': 'ReLU',
            'globalActivation': 'ReLU',
        }.get(name, choices[0]))
        
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: {
            'hiddenDim': 64,
            'numLayers': 2,
            'globalEmbeddingDim': 32,
            'globalNumLayers': 1,
        }.get(name, 32))
        
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high: {
            'recurrentDropout': 0.2,
            'globalDropout': 0.1,
        }.get(name, 0.1))
        
        archParams = {}
        datasetInfo = {
            'timeSeriesShape': (34, 160),
            'globalFeaturesShape': (32,),
        }
        
        result = pipeline.setUpEncoder(mock_trial, archParams, datasetInfo)
        
        assert 'EncoderParams' in result
        assert 'GlobalFFEncoderParams' in result
        assert result['EncoderParams']['layer_type'][0]['name'] == 'LSTM'
    
    def test_setUp_encoder_conv1d(self, basic_params):
        """Test setUpEncoder with Conv1d architecture."""
        pipeline = FinalPipeline(basic_params)
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(side_effect=lambda name, choices: {
            'architectureType': 'Conv1d',
            'kernelSize': 3,
            'strideConv1D': 1,
            'encoderActivation': 'ReLU',
            'globalActivation': 'ReLU',
            'poolType_0': 'max',
        }.get(name, choices[0]))
        
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: {
            'numConvLayers': 2,
            'convChannels_0': 64,
            'convChannels_1': 128,
            'globalEmbeddingDim': 32,
            'globalNumLayers': 1,
        }.get(name, 32))
        
        mock_trial.suggest_float = Mock(return_value=0.1)
        
        archParams = {}
        datasetInfo = {
            'timeSeriesShape': (34, 160),
            'globalFeaturesShape': (32,),
        }
        
        result = pipeline.setUpEncoder(mock_trial, archParams, datasetInfo)
        
        assert 'EncoderParams' in result
        assert result['EncoderParams']['layer_type'][0]['name'] == 'Conv1d'
    
    def test_setUp_feedforward_head(self, basic_params):
        """Test setUpFeedForwardHead."""
        pipeline = FinalPipeline(basic_params)
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_categorical = Mock(return_value='ReLU')
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high: {
            'numFFLayers': 2,
            'ffHiddenDim': 64,
        }.get(name, 32))
        mock_trial.suggest_float = Mock(return_value=0.2)
        
        archParams = {
            'EncoderParams': {
                'activation_function': 'ReLU',
                'layer_type': [{
                    'name': 'LSTM',
                    'params': {
                        'hidden_size': 64,
                        'bidirectional': False
                    }
                }]
            },
            'GlobalFFEncoderParams': {
                'activation_function': 'ReLU',
                'layer_type': [{
                    'name': 'Linear',
                    'params': {
                        'out_features': 32
                    }
                }]
            }
        }
        
        datasetInfo = {}
        
        result = pipeline.setUpFeedForwardHead(mock_trial, archParams, datasetInfo)
        
        assert 'FeedForwardParams' in result
        assert 'layer_type' in result['FeedForwardParams']
        assert len(result['FeedForwardParams']['layer_type']) > 0


class TestHelperMethods:
    """Test helper methods."""
    
    def test_apply_he_initialization_linear(self, basic_params):
        """Test He initialization on Linear layers."""
        pipeline = FinalPipeline(basic_params)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )
        
        # Apply initialization
        pipeline.apply_he_initialization(model, activation_type="ReLU")
        
        # Check that weights are not all zeros (initialized)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert not torch.all(module.weight == 0)
    
    def test_apply_he_initialization_conv1d(self, basic_params):
        """Test He initialization on Conv1d layers."""
        pipeline = FinalPipeline(basic_params)
        
        model = torch.nn.Sequential(
            torch.nn.Conv1d(34, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=3)
        )
        
        # Apply initialization
        pipeline.apply_he_initialization(model, activation_type="ReLU")
        
        # Check that weights are not all zeros
        for module in model.modules():
            if isinstance(module, torch.nn.Conv1d):
                assert not torch.all(module.weight == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
