# Test Suite Documentation

This document provides a comprehensive overview of all test classes and their test methods in the project.

**Overall Coverage: 90%**  
**Total Tests: 293**  
**Status: All Passing ✅**

---

## Table of Contents
- [Architecture Tests](#architecture-tests)
  - [Decoder Tests](#decoder-tests)
  - [Direct Tests](#direct-tests)
  - [Encoder Tests](#encoder-tests)
  - [FeedForward Tests](#feedforward-tests)
  - [LightningAutoencoder Tests](#lightningautoencoder-tests)
  - [ParameterInterpreter Tests](#parameterinterpreter-tests)
- [Pipeline Tests](#pipeline-tests)
  - [DataLoader Tests](#dataloader-tests)
  - [OptunaOptimizer Tests](#optunaoptimizer-tests)
  - [Pipeline Tests](#pipeline-tests-1)
  - [SubmissionGenerator Tests](#submissiongenerator-tests)
- [PreProcessing Tests](#preprocessing-tests)

---


## Run All Tests
```bash
poetry run pytest tests/
```
```
poetry run pytest --cov=src --cov-report=term-missing tests/
```


---

## Architecture Tests

### Decoder Tests
**File:** `ArchitectureTest/test_Decoder.py`  
**Coverage:** 98%  
**Test Classes:** 11

#### TestDecoderBasic
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_decoder_initialization` | Tests basic decoder initialization | Decoder instance created, has `net` attribute |
| `test_decoder_forward` | Tests forward pass with sample input | Output tensor shape matches expected dimensions |
| `test_decoder_linear_layers` | Tests decoder with linear layers only | Correct layer types in network |
| `test_decoder_conv_transpose_layers` | Tests decoder with Conv2d transpose layers | Upsampling works correctly |
| `test_decoder_mixed_layers` | Tests decoder with mixed layer types | All layer types processed correctly |

#### TestDecoderActivations
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_decoder_relu_activation` | Tests ReLU activation function | ReLU layers present in network |
| `test_decoder_gelu_activation` | Tests GELU activation function | GELU layers present in network |
| `test_decoder_leaky_relu_activation` | Tests LeakyReLU activation | LeakyReLU layers present in network |
| `test_decoder_no_activation_on_last_layer` | Tests no activation after final layer | No activation after last layer |
| `test_decoder_output_activation_sigmoid` | Tests Sigmoid output activation | Output values in [0, 1] range |
| `test_decoder_output_activation_tanh` | Tests Tanh output activation | Output values in [-1, 1] range |

#### TestDecoderAdvanced
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_decoder_batch_norm` | Tests BatchNorm layers | BatchNorm present in network |
| `test_decoder_dropout` | Tests Dropout layers | Dropout layers added correctly |
| `test_decoder_flatten_unflatten` | Tests Flatten/Unflatten operations | Shape transformations correct |
| `test_decoder_multiple_conv_layers` | Tests multiple convolutional layers | Multiple Conv layers stacked correctly |

#### TestDecoderRecurrent
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_decoder_lstm_basic` | Tests basic LSTM decoder | LSTM processes sequences correctly |
| `test_decoder_gru` | Tests GRU decoder | GRU output shape correct |
| `test_decoder_rnn` | Tests vanilla RNN decoder | RNN processes input correctly |
| `test_decoder_bidirectional_lstm` | Tests bidirectional LSTM | Hidden size doubled for bidirectional |
| `test_decoder_lstm_with_linear` | Tests LSTM + Linear combination | Both layer types work together |
| `test_decoder_activation_skip_after_recurrent` | Tests activation skipping after RNN | Activation not applied after GRU |

#### Additional Test Classes
- **TestDecoderEdgeCases**: Empty params, different batch sizes, gradient flow
- **TestDecoderComplexArchitectures**: Multi-layer decoders, residual connections
- **TestDecoderOutputShapes**: Various input/output dimension combinations
- **TestDecoderParameterCounts**: Verifies learnable parameters exist
- **TestDecoderEvalMode**: Tests evaluation mode behavior
- **TestDecoderDeviceHandling**: CPU/GPU device compatibility
- **TestDecoderRecurrentAdvanced**: Advanced LSTM/GRU configurations

---

### Direct Tests
**File:** `ArchitectureTest/test_Direct.py`  
**Coverage:** 100% 🌟  
**Test Class:** TestDirect

| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_direct_initialization` | Tests Direct model initialization | Model created with all components |
| `test_direct_forward` | Tests forward pass | Output shape correct with zero column |
| `test_direct_training_step` | Tests training step with labels | Loss computed correctly |
| `test_direct_training_step_unlabeled` | Tests training with no labels | Loss = 0 for unlabeled data |
| `test_direct_validation_step_basic` | Tests validation step | Validation metrics logged |
| `test_direct_with_class_weights` | Tests loading class weights from YAML | Weights loaded and applied |
| `test_direct_with_invalid_class_weights_path` | Tests invalid weights file path | Error handled gracefully |
| `test_direct_training_step_with_class_weights` | Tests training uses class weights | Weighted loss computed |
| `test_direct_on_validation_epoch_end` | Tests epoch end callback | F1 metric reset called |
| `test_direct_configure_optimizers` | Tests optimizer configuration | Returns optimizer + lr_scheduler dict |
| `test_direct_encoder_output_shape` | Tests encoder output dimensions | Correct latent dimension |
| `test_direct_feedforward_output_shape` | Tests feedforward output | Correct output dimension + 1 |
| `test_direct_has_f1_metric` | Tests F1 metric initialization | val_f1 attribute exists |
| `test_direct_forward_adds_zero_column` | Tests zero column addition | Last column is zeros |

---

### Encoder Tests
**File:** `ArchitectureTest/test_Encoder.py`  
**Coverage:** 91%  
**Test Classes:** 10

#### TestEncoderBasic
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_encoder_initialization` | Tests basic encoder initialization | Encoder instance created |
| `test_encoder_forward` | Tests forward pass | Output tensor correct shape |
| `test_encoder_conv2d_layers` | Tests 2D convolutional layers | Conv2d layers in network |
| `test_encoder_linear_layers` | Tests linear layers only | Linear transformation works |
| `test_encoder_mixed_layers` | Tests mixed layer types | All layer types integrated |

#### TestEncoderActivations
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_encoder_relu_activation` | Tests ReLU activation | ReLU present after layers |
| `test_encoder_gelu_activation` | Tests GELU activation | GELU applied correctly |
| `test_encoder_leaky_relu_activation` | Tests LeakyReLU activation | LeakyReLU in network |

#### TestEncoderRecurrent
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_encoder_lstm` | Tests LSTM encoder | LSTM processes sequences, last timestep extracted |
| `test_encoder_gru` | Tests GRU encoder | GRU output shape correct |
| `test_encoder_rnn` | Tests vanilla RNN | RNN processes input |
| `test_encoder_bidirectional_lstm` | Tests bidirectional LSTM | Hidden size doubled |
| `test_encoder_lstm_multilayer` | Tests multi-layer LSTM | Multiple LSTM layers stacked |
| `test_encoder_activation_after_recurrent_layer` | Tests activation skipping | No activation applied after LSTM |

#### Additional Test Classes
- **TestEncoderAdvanced**: BatchNorm, Dropout, MaxPooling
- **TestEncoderEdgeCases**: Empty params, different batch sizes
- **TestEncoderComplexArchitectures**: Deep encoders, wide encoders
- **TestEncoderOutputShapes**: Various input/output combinations
- **TestEncoderGradientFlow**: Gradient backpropagation
- **TestEncoderParameterCount**: Learnable parameters exist
- **TestEncoderRecurrentAdvanced**: Advanced RNN configurations

---

### FeedForward Tests
**File:** `ArchitectureTest/test_FeedForward.py`  
**Coverage:** 100% 🌟  
**Test Classes:** 6

#### TestFeedForwardBasic
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_feedforward_initialization` | Tests basic initialization | Network created successfully |
| `test_feedforward_forward` | Tests forward pass | Output shape correct |
| `test_feedforward_single_linear_layer` | Tests single layer network | Single Linear layer works |
| `test_feedforward_multiple_linear_layers` | Tests multi-layer network | Multiple layers stacked |

#### TestFeedForwardActivations
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_feedforward_relu_activation` | Tests ReLU activation | ReLU between layers |
| `test_feedforward_gelu_activation` | Tests GELU activation | GELU applied correctly |
| `test_feedforward_leaky_relu_activation` | Tests LeakyReLU activation | LeakyReLU in network |
| `test_feedforward_sigmoid_activation` | Tests Sigmoid activation | Sigmoid used as activation |
| `test_feedforward_tanh_activation` | Tests Tanh activation | Tanh applied between layers |

#### Additional Test Classes
- **TestFeedForwardAdvanced**: BatchNorm, Dropout, complex architectures
- **TestFeedForwardEdgeCases**: Empty params, different batch sizes
- **TestFeedForwardOutputActivation**: Specific output layer activations
- **TestFeedForwardGradientFlow**: Backpropagation verification

---

### LightningAutoencoder Tests
**File:** `ArchitectureTest/test_LightningAutoencoder.py`  
**Coverage:** 85%  
**Test Classes:** 4

#### TestLightningAutoencoderInitialization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_autoencoder_split_initialization` | Tests split architecture init | Encoder and decoder created |
| `test_autoencoder_joint_initialization` | Tests joint architecture init | Shared encoder/decoder |
| `test_autoencoder_has_components` | Tests all components present | Encoder, decoder, FF present |

#### TestLightningAutoencoderForward
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_autoencoder_forward_split` | Tests forward pass (split) | Reconstruction and prediction correct |
| `test_autoencoder_forward_joint` | Tests forward pass (joint) | Joint processing works |

#### TestLightningAutoencoderTraining
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_training_step_labeled` | Tests training with labels | Loss computed for labeled data |
| `test_training_step_unlabeled` | Tests training without labels | Reconstruction loss only |
| `test_validation_step` | Tests validation step | Validation metrics computed |

#### TestLightningAutoencoderOptimizer
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_configure_optimizers` | Tests optimizer setup | Adam optimizer with LR scheduler |

---

### ParameterInterpreter Tests
**File:** `ArchitectureTest/test_ParameterInterpreter.py`  
**Coverage:** 100% 🌟  
**Test Classes:** 5

#### TestParameterInterpreterBasic
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_initialization` | Tests basic initialization | Interpreter created with mappings |
| `test_interpret_valid_key` | Tests valid key interpretation | Correct class returned |
| `test_interpret_invalid_key` | Tests invalid key handling | KeyError raised |

#### TestParameterInterpreterRequiredParams
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_check_required_params_valid` | Tests valid required params | No error raised |
| `test_check_required_params_missing` | Tests missing required param | KeyError raised |
| `test_check_required_params_invalid_value` | Tests invalid param value | ValueError raised |

#### Additional Test Classes
- **TestParameterInterpreterMultipleKeys**: Multiple architecture types
- **TestParameterInterpreterComplexRequirements**: Complex validation rules
- **TestParameterInterpreterEdgeCases**: Empty dicts, special cases

---

## Pipeline Tests

### DataLoader Tests
**File:** `test_DataLoader.py`  
**Coverage:** 87%  
**Test Class:** TestDataModule

| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_dataloader_initialization` | Tests DataModule initialization | Parameters set correctly |
| `test_dataloader_missing_required_params` | Tests missing params error | KeyError raised |
| `test_dataloader_setup` | Tests dataset setup | Train/val/test datasets created |
| `test_dataloader_train_dataloader` | Tests train DataLoader creation | Returns valid DataLoader |
| `test_dataloader_val_dataloader` | Tests validation DataLoader | Returns valid DataLoader |
| `test_dataloader_test_dataloader` | Tests test DataLoader | Returns valid DataLoader |
| `test_dataloader_predict_dataloader` | Tests predict DataLoader | Same as test DataLoader |
| `test_dataloader_without_setup_raises_error` | Tests error without setup | RuntimeError/AttributeError raised |
| `test_dataloader_cuda_pin_memory` | Tests pin_memory with CUDA | pin_memory=True when CUDA available |

---

### OptunaOptimizer Tests
**File:** `test_OptunaOptimizer.py`  
**Coverage:** 84%  
**Test Classes:** 3

#### TestOptunaOptimizerInitialization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_optimizer_initialization` | Tests basic initialization | Optimizer created with config |
| `test_optimizer_with_database` | Tests database storage | PostgreSQL storage configured |

#### TestOptunaOptimizerSuggest
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_suggest_categorical` | Tests categorical parameter suggestion | Correct value from choices |
| `test_suggest_float` | Tests float parameter suggestion | Value in range |
| `test_suggest_int` | Tests integer parameter suggestion | Integer in range |
| `test_suggest_mixed_params` | Tests mixed parameter types | All types handled correctly |

#### TestOptunaOptimizerOptimization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_optimize_simple` | Tests optimization process | Best parameters found |
| `test_optimize_with_trials` | Tests multiple trials | N trials executed |

---

### Pipeline Tests
**File:** `test_Pipeline.py`  
**Coverage:** 88%  
**Test Class:** TestPipeline (implied)

| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_constructor_with_dict_config` | Tests Pipeline constructor | DataLoader initialized |
| `test_constructor_creates_datamodule` | Tests DataModule creation | DataModule instance created |
| `test_build_architecture_autoencoder_split` | Tests split autoencoder building | LightningAutoencoder returned |
| `test_build_architecture_autoencoder_joint` | Tests joint autoencoder building | LightningAutoencoder returned |
| `test_build_architecture_direct` | Tests Direct architecture building | Direct instance returned |
| `test_build_architecture_invalid_type` | Tests invalid architecture type | KeyError raised |
| `test_pipeline_has_fit_and_validate` | Tests fit_and_validate exists | Method callable |
| `test_build_architecture_different_output_dims` | Tests different output dimensions | Architectures built with various dims |
| `test_build_architecture_missing_arch_type` | Tests missing arch_type | KeyError raised |
| `test_pipeline_multiple_builds` | Tests building multiple architectures | Different instances created |
| `test_fit_and_validate_has_correct_signature` | Tests method signature | Correct parameters present |
| `test_fit_and_validate_builds_architecture` | Tests architecture building in fit | build_architecture called |
| `test_fit_and_validate_configures_loader` | Tests loader configuration | setup() called with params |

---

### SubmissionGenerator Tests
**File:** `test_SubmissionGenerator.py`  
**Coverage:** 92%  
**Test Classes:** 4

#### TestSubmissionGeneratorInitialization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_initialization` | Tests basic initialization | Generator created with params |
| `test_initialization_missing_params` | Tests missing required params | KeyError raised |

#### TestSubmissionGeneratorLabelMapping
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_map_predictions_to_labels` | Tests numeric to label mapping | Correct string labels returned |
| `test_map_predictions_batch` | Tests batch label mapping | All predictions mapped |

#### TestSubmissionGeneratorFileGeneration
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_generate_submission_file` | Tests submission file creation | CSV file created with correct format |
| `test_generate_submission_file_content` | Tests file content correctness | Correct sample_id and label columns |

#### TestSubmissionGeneratorEdgeCases
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_empty_predictions` | Tests empty prediction array | Handles empty input |
| `test_large_batch_predictions` | Tests large prediction batches | Handles large datasets |

---

## PreProcessing Tests

### PreProcessor Tests
**File:** `test_PreProcessor.py`  
**Coverage:** 89%  
**Test Classes:** 8

#### TestPreProcessorInit
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_init_with_basic_params` | Tests basic param initialization | Parameters set correctly |
| `test_init_with_default_values` | Tests default parameter values | Defaults applied correctly |
| `test_init_with_pca_params` | Tests PCA parameter initialization | PCA config set |
| `test_init_with_feature_selection` | Tests feature selection params | Feature selection configured |
| `test_from_yaml` | Tests loading from YAML file | Params loaded from YAML |

#### TestPreProcessorDataHandling
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_load_data` | Tests CSV data loading | DataFrame loaded correctly |
| `test_save_data_dataframe` | Tests saving DataFrame | CSV file created |
| `test_save_data_numpy_array` | Tests saving NumPy array | Array saved as CSV |
| `test_save_data_creates_directory` | Tests directory creation | Nested directories created |

#### TestPreProcessorTransformations
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_remove_last_column` | Tests last column removal | Column removed correctly |
| `test_handle_is_pirate_default` | Tests default isPirate handling | isPirate column created from n_eyes |
| `test_handle_is_pirate_with_one_hot_encoding` | Tests one-hot encoding | isPirate and isNotPirate created |
| `test_handle_is_pirate_drop_all` | Tests dropping all pirate features | All pirate columns removed |

#### TestPreProcessorNormalization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_normalize_basic` | Tests basic normalization | Mean=0, std=1 for training |
| `test_normalize_with_exclusions` | Tests column exclusions | Excluded columns unchanged |
| `test_normalize_with_zero_std` | Tests constant feature handling | Constant features centered only |

#### TestPreProcessorFeatureSelection
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_apply_feature_selection` | Tests feature selection | Only selected features kept |

#### TestPreProcessorClassWeights
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_compute_and_save_class_weights` | Tests class weight computation | Weights inversely proportional to frequency |
| `test_compute_class_weights_balanced` | Tests balanced class weights | Equal weights for balanced classes |

#### TestPreProcessorPCA
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_apply_pca_basic` | Tests basic PCA application | Dimensionality reduced correctly |
| `test_apply_pca_preserves_pirate_columns` | Tests column preservation | isPirate columns kept after PCA |
| `test_apply_pca_with_verbose` | Tests PCA with plotting | Verbose mode plots generated |

#### TestPreProcessorIntegration
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_preprocess_basic_workflow` | Tests full preprocessing pipeline | All steps executed successfully |
| `test_preprocess_with_pca` | Tests preprocessing with PCA | PCA applied in pipeline |
| `test_preprocess_with_feature_selection` | Tests preprocessing with selection | Feature selection applied |
| `test_preprocess_handles_load_error` | Tests load error handling | Errors caught and reported |
| `test_preprocess_handles_column_removal_error` | Tests column removal errors | Errors handled gracefully |
| `test_preprocess_handles_normalization_error` | Tests normalization errors | Inf/NaN values handled |
| `test_preprocess_handles_pca_error` | Tests PCA error handling | Minimal data errors caught |
| `test_preprocess_handles_feature_selection_error` | Tests selection errors | Invalid features handled |
| `test_preprocess_handles_save_error` | Tests save error handling | Platform-specific error handling |

#### TestPreProcessorVisualization
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_plot_one_time_series` | Tests time series plotting | Plots generated without errors |

#### TestPreProcessorEdgeCases
| Test Method | Description | Key Assertions |
|------------|-------------|----------------|
| `test_handle_is_pirate_with_unknown_eye_value` | Tests unknown n_eyes values | Unknown mapped to 0 |
| `test_normalize_with_nan_values` | Tests NaN handling | NaN values handled in normalization |
| `test_apply_pca_single_sample` | Tests PCA with one sample | Single sample handled or error caught |
| `test_preprocess_saves_labels_unchanged` | Tests label preservation | Labels saved without modification |
| `test_remove_last_column_single_column` | Tests single column removal | Empty DataFrame returned |

---
