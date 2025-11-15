# Pipeline Module

## Description
The Pipeline module provides a production-ready framework for end-to-end deep learning workflows with automated hyperparameter optimization using K-fold cross-validation. The **FinalPipeline** class orchestrates architecture search, distributed training across folds, model checkpointing, and submission generation. It integrates Optuna with TPE sampling and MedianPruner for efficient hyperparameter search, PostgreSQL for persistent storage, PyTorch Lightning for training, and TensorBoard for logging. The pipeline supports multiple architecture types (Direct, Autoencoder), encoder variants (LSTM/GRU, 1D-CNN, Multi-Scale CNN), model-level windowing, and automatic He initialization for robust model training.

---

## Main Classes

### `FinalPipeline`
Production-ready orchestration class managing the complete machine learning workflow with K-fold cross-validation, automated hyperparameter optimization, and persistent experiment tracking.

**Configuration Parameters:**
- `project_name` (str, **required**): Project identifier for experiment organization
- `study_name` (str, **required**): Unique study name for Optuna optimization
- `database_url` (str, **required**): PostgreSQL connection URL for persistent storage
- `data_params` (dict, **required**): DataModule configuration including:
  - `train_path`: Path to training time series CSV
  - `test_path`: Path to test time series CSV
  - `train_labels_path`: Path to training labels CSV
  - `train_global_path`: Path to training global features CSV
  - `test_global_path`: Path to test global features CSV
  - `class_weights_path`: Path to class weights YAML
  - `n_folds`: Number of K-fold cross-validation folds
  - `batch_size`: Training batch size
  - `num_workers`: DataLoader worker processes
- `submission_path` (str, **required**): Directory for saving submission CSV files

**Core Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `params: dict` | - | Initialize FinalPipeline with configuration, database connection, and Optuna study. |
| `optuna_optimize` | `n_trials: int` | - | Run Optuna optimization with K-fold cross-validation for specified number of trials. |
| `study_summary` | - | - | Print comprehensive study statistics including best trial, top 5 trials, and recent states. |
| `load_best_model` | - | `torch.nn.Module` | Load best model from study checkpoints with automatic architecture reconstruction and windowing. |
| `create_submission` | - | `pd.DataFrame` | Generate competition submission CSV using best model on test data. |
| `objective_kfold` | `trial: optuna.Trial` | `float` | K-fold cross-validation objective function returning mean validation F1-score. |

**Architecture Setup Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `setUpEncoder` | `trial: optuna.Trial`<br>`architectureParameters: dict`<br>`datasetInfo: dict` | `dict` | Configure encoder architecture (Recurrent/Conv1d/MultiScaleCNN) with hyperparameter suggestions. |
| `setUpDecoder` | `trial: optuna.Trial`<br>`architectureParameters: dict`<br>`datasetInfo: dict`<br>`encoderParams: dict` | `dict` | Configure symmetric decoder architecture for autoencoder models. |
| `setUpFeedForwardHead` | `trial: optuna.Trial`<br>`architectureParameters: dict`<br>`datasetInfo: dict` | `dict` | Configure classification feedforward head with dropout and layer configurations. |
| `apply_he_initialization` | `model: nn.Module`<br>`activation_type: str` | - | Apply He initialization to all Linear/Conv layers based on activation function. |

**Supported Configurations:**

**Macro Architectures:**
- `Direct`: Direct classification optimizing only cross-entropy loss
- `Autoencoder`: Joint reconstruction and classification with multi-task loss

**Encoder Types:**
- `Recurrent`: LSTM/GRU with configurable layers, hidden dimensions, bidirectionality
- `Conv1d`: 1D convolutional layers with pooling for sequence processing
- `MultiScaleCNN`: Inception-style multi-scale convolutions with parallel branches

**Windowing:**
- Model-level windowing via `WindowedModelWrapper`
- Configurable window size [5-40], stride ratio [0.25-1.0]
- Aggregation methods: avg_probs, avg_logits, majority_vote, max_confidence

**Learning Rate Schedulers:**
- `ReduceLROnPlateau`: Adaptive LR reduction on validation plateau
- `CosineAnnealing`: Smooth cosine decay over training
- `CosineAnnealingWarmRestarts`: Periodic restarts for escaping local minima

---

## FinalPipeline Architecture

![Architecture Diagram](../../../Deliverables/UML/UML_drawio.png)

```
┌───────────────────────────────────────────────────────────────────────┐
│                          FinalPipeline                                │
│                                                                       │
│  ┌────────────────┐      ┌──────────────────┐      ┌──────────────┐  │
│  │  PostgreSQL    │◀────▶│  Optuna Study    │◀────▶│ TPE Sampler  │  │
│  │  Storage       │      │  - TPESampler    │      │ + Pruner     │  │
│  │  - Persistent  │      │  - MedianPruner  │      └──────────────┘  │
│  │  - Resumable   │      │  - Seed: 42      │                         │
│  └────────────────┘      └──────────────────┘                         │
│         │                         │                                    │
│         │                         ▼                                    │
│         │              ┌─────────────────────┐                         │
│         │              │  objective_kfold    │                         │
│         │              │  - Suggest params   │                         │
│         │              │  - K-fold CV        │                         │
│         │              └─────────────────────┘                         │
│         │                         │                                    │
│         │                         ▼                                    │
│         │         ┌───────────────────────────────┐                    │
│         │         │   Architecture Builder        │                    │
│         │         │   - setUpEncoder()            │                    │
│         │         │   - setUpDecoder()            │                    │
│         │         │   - setUpFeedForwardHead()    │                    │
│         │         │   - apply_he_initialization() │                    │
│         │         └───────────────────────────────┘                    │
│         │                         │                                    │
│         │                         ▼                                    │
│         │         ┌───────────────────────────────┐                    │
│         │         │  K-Fold Training Loop         │                    │
│         │         │  For fold k=1 to K:           │                    │
│         │         │    - DataModule.setup_fold(k) │                    │
│         │         │    - Create model instance    │                    │
│         │         │    - WindowedModelWrapper     │                    │
│         │         │    - PyTorch Lightning Train  │                    │
│         │         │    - EarlyStopping + Checkpt  │                    │
│         │         │    - TensorBoard logging      │                    │
│         │         └───────────────────────────────┘                    │
│         │                         │                                    │
│         │                         ▼                                    │
│         │         ┌───────────────────────────────┐                    │
│         │         │  Return mean(F1_scores)       │                    │
│         │         │  Store fold scores & std      │                    │
│         └────────▶│  Save checkpoints to disk     │                    │
│                   └───────────────────────────────┘                    │
│                                  │                                     │
│                                  ▼                                     │
│                   ┌───────────────────────────────┐                    │
│                   │  load_best_model()            │                    │
│                   │  - Load from checkpoint       │                    │
│                   │  - Reconstruct architecture   │                    │
│                   │  - Apply windowing wrapper    │                    │
│                   └───────────────────────────────┘                    │
│                                  │                                     │
│                                  ▼                                     │
│                   ┌───────────────────────────────┐                    │
│                   │  create_submission()          │                    │
│                   │  - WindowedSubmissionGen      │                    │
│                   │  - Aggregate predictions      │                    │
│                   │  - Save CSV                   │                    │
│                   └───────────────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Workflow

### 1. **Initialization**
```python
pipeline = FinalPipeline(params={
    "project_name": "PiratePain",
    "study_name": "_experiment_v1",
    "database_url": "postgresql://user:password@localhost:5432/optuna_db",
    "data_params": {...},
    "submission_path": "./submissions"
})
```
- Creates/loads Optuna study from PostgreSQL
- Configures TPE sampler (seed=42) and MedianPruner
- Initializes database connection for persistent storage

### 2. **Hyperparameter Optimization**
```python
pipeline.optuna_optimize(n_trials=100)
```
For each trial:
1. **Suggest hyperparameters**: Architecture type, encoder config, learning rate, etc.
2. **Create DataModule**: Setup K-fold stratified splits
3. **Build architecture**: Encoder + Decoder (if AE) + Classification head
4. **K-fold training**:
   - Train separate model on each fold
   - Apply He initialization based on activation
   - Wrap with WindowedModelWrapper if windowing enabled
   - EarlyStopping + ModelCheckpoint callbacks
   - TensorBoard logging per fold
5. **Aggregate results**: Return mean F1-score across folds
6. **Prune if needed**: MedianPruner stops unpromising trials early

### 3. **Study Monitoring**
```python
pipeline.study_summary()
```
- Display best trial and F1-score
- Show top 5 trials with architecture details
- List recent trial states (✓/✗/⊗ symbols)

### 4. **Model Loading**
```python
best_model = pipeline.load_best_model()
```
- Retrieve best trial hyperparameters
- Locate best checkpoint among all folds
- Reconstruct architecture configuration
- Load model weights from checkpoint
- Apply WindowedModelWrapper if used in training
- Set to eval mode and move to device

### 5. **Submission Generation**
```python
submission_df = pipeline.create_submission()
```
- Use best model for test set inference
- WindowedSubmissionGenerator handles window aggregation
- Map predictions to class labels
- Save CSV with zero-padded sample indices

---

## Module Components

### 1. **DataLoader**
Handles multimodal data loading with K-fold stratified cross-validation.
- Time series: (batch, 34 features, 160 timesteps)
- Global features: (batch, d_g dimensions)
- Supports windowing at data level (disabled in FinalPipeline)
- K-fold splits with optional test data inclusion for autoencoders

### 2. **Architectures**
Modular building blocks and complete models:
- **Encoder**: LSTM/GRU, Conv1d, MultiScaleCNN
- **Decoder**: Symmetric architecture for reconstruction
- **FeedForward**: Classification head with dropout
- **Direct**: Classification-only architecture
- **LightningAutoencoder**: Multi-task learning (classification + reconstruction)
- **WindowedModelWrapper**: Model-level windowing with aggregation

### 3. **Optimizer**
Optuna-based hyperparameter optimization:
- **TPESampler**: Bayesian optimization with kernel density estimates
- **MedianPruner**: Early stopping of unpromising trials
- **PostgreSQL backend**: Persistent, distributed, fault-tolerant storage
- **Search space**: >10^15 configurations

### 4. **SubmissionGenerator**
CSV generation for competition submissions:
- **WindowedSubmissionGenerator**: Handles windowed model predictions
- **Aggregation methods**: avg_probs, avg_logits, majority_vote, max_confidence
- **Label mapping**: Integer → {no_pain, low_pain, high_pain}
- **Format**: Zero-padded sample indices

### 5. **Utils**
Support utilities:
- **ParameterInterpreter**: Validate architecture configurations
- **EnsembleModels**: Model ensembling (optional)
- **FeatureSelector**: Supervised feature selection for preprocessing
- **He initialization**: Automatic weight initialization based on activation
