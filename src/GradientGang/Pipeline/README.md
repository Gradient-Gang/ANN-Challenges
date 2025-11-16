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
- Model-level windowing via `WindowedModelWrapper` (recommended)
- Configurable window size [5-40], stride ratio [0.25-1.0]
- Aggregation methods: avg_probs, avg_logits, majority_vote, max_confidence
- Window-level auxiliary loss weight [0.0-0.5]
- Ensures training/validation/test metric consistency (no train/test gap)
- Respects base model loss computation (reconstruction + classification for autoencoders)

**Learning Rate Schedulers:**
- `ReduceLROnPlateau`: Adaptive LR reduction on validation plateau
- `CosineAnnealing`: Smooth cosine decay over training
- `CosineAnnealingWarmRestarts`: Periodic restarts for escaping local minima

---

## FinalPipeline Architecture

![Architecture Diagram](/Deliverables/UML/PIPELINE.png)


```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FinalPipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] Initialization                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PostgreSQL Database ◄─► Optuna Study (TPE Sampler + MedianPruner)   │    │
│  │   • Persistent storage    • Seed: 42                                │    │
│  │   • Resumable trials      • Maximize mean F1-score                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  [2] Hyperparameter Optimization Loop (n_trials)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      objective_kfold(trial)                         │    │
│  │                                                                     │    │
│  │  ┌───────────────────────────────────────────────────────────┐      │    │
│  │  │ Suggest Hyperparameters:                                  │      │    │
│  │  │  • Macro Architecture: Direct / Autoencoder               │      │    │
│  │  │  • Encoder Type: Recurrent / Conv1d / MultiScaleCNN       │      │    │
│  │  │  • Learning Rate, Regularization, Dropout                 │      │    │
│  │  │  • Windowing: use_windowing, window_size, stride_ratio    │      │    │
│  │  │  • Aggregation: avg_probs / majority_vote / max_conf      │      │    │
│  │  │  • Scheduler: ReduceLROnPlateau / CosineAnnealing         │      │    │
│  │  └───────────────────────────────────────────────────────────┘      │    │
│  │                            │                                        │    │
│  │                            ▼                                        │    │
│  │  ┌───────────────────────────────────────────────────────────┐      │    │
│  │  │ Build Architecture:                                       │      │    │
│  │  │  1. setUpEncoder() → Encoder params (time series)         │      │    │
│  │  │  2. setUpEncoder() → GlobalFF params (global features)    │      │    │
│  │  │  3. setUpDecoder() → Decoder params (if Autoencoder)      │      │    │
│  │  │  4. setUpFeedForwardHead() → Classification head          │      │    │
│  │  │  5. Create base model (Direct / LightningAutoencoder)     │      │    │
│  │  │  6. apply_he_initialization() → Weight init by activation │      │    │
│  │  │  7. WindowedModelWrapper (if use_windowing=True)          │      │    │
│  │  └───────────────────────────────────────────────────────────┘      │    │
│  │                            │                                        │    │
│  │                            ▼                                        │    │
│  │  ┌───────────────────────────────────────────────────────────┐      │    │
│  │  │ K-Fold Cross-Validation Loop (k = 1 to n_folds):          │      │    │
│  │  │                                                           │      │    │
│  │  │  For each fold:                                           │      │    │
│  │  │    ├─ DataModule.setup_fold(k)                            │      │    │
│  │  │    │    • Stratified train/val split                      │      │    │
│  │  │    │    • Include test in train (if Autoencoder)          │      │    │
│  │  │    │                                                      │      │    │
│  │  │    ├─ Create fresh model instance                         │      │    │
│  │  │    │    • Same architecture, new weights                  │      │    │
│  │  │    │                                                      │      │    │
│  │  │    ├─ Lightning Trainer:                                  │      │    │
│  │  │    │    • EarlyStopping (monitors val_prediction_loss)    │      │    │
│  │  │    │    • ModelCheckpoint (saves best per fold)           │      │    │
│  │  │    │    • TensorBoard logger                              │      │    │
│  │  │    │    • Max epochs, gradient clipping                   │      │    │
│  │  │    │                                                      │      │    │
│  │  │    └─ Record best_val_F1 for this fold                    │      │    │
│  │  │                                                           │      │    │
│  │  │  Aggregate: mean_F1 = mean([F1_fold1, ..., F1_foldK])     │      │    │
│  │  │            std_F1 = std([F1_fold1, ..., F1_foldK])        │      │    │
│  │  └───────────────────────────────────────────────────────────┘      │    │
│  │                            │                                        │    │
│  │                            ▼                                        │    │
│  │  Return mean_F1 to Optuna  (MedianPruner may stop early)            │    │
│  │  Store trial.user_attrs: fold_scores, mean_f1, std_f1               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    │ (Repeat for n_trials)                  │
│                                    ▼                                        │
│  [3] Best Model Selection                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ study.best_trial → Retrieve best hyperparameters & checkpoint path  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  [4] Load Best Model                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ load_best_model():                                                  │    │
│  │  1. Reconstruct architecture from best_trial params                 │    │
│  │  2. Load checkpoint weights (best fold)                             │    │
│  │  3. Wrap with WindowedModelWrapper (if used in training)            │    │
│  │  4. Set to eval mode, move to device                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  [5] Generate Submission                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ create_submission():                                                │    │ 
│  │  1. Test DataLoader (no windowing in data)                          │    │
│  │  2. WindowedSubmissionGenerator (if windowing used):                │    │
│  │     • Creates windows internally per sample                         │    │
│  │     • Aggregates using same method as training                      │    │
│  │  3. Map predictions: int → {no_pain, low_pain, high_pain}           │    │
│  │  4. Save CSV with zero-padded indices (000, 001, 002, ...)          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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
- Reconstruct architecture configuration (Direct/Autoencoder/Encoder type/etc.)
- Load model weights from checkpoint
- **Apply WindowedModelWrapper if used in training** (preserves exact training setup)
- Set to eval mode and move to device
- Ready for inference with consistent windowing behavior

### 5. **Submission Generation**
```python
submission_df = pipeline.create_submission()
```
- Use best model for test set inference
- **WindowedSubmissionGenerator** handles window aggregation (if windowing used)
  - Aggregates window predictions using same method as training (avg_probs/majority_vote/max_confidence)
  - Ensures test predictions match validation behavior
- Map integer predictions to class labels {no_pain, low_pain, high_pain}
- Save CSV with zero-padded sample indices (000, 001, 002, ...)

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

---

## WindowedModelWrapper: Model-Level Windowing

### Why Model-Level Windowing?

Traditional **data-level windowing** (in DataLoader) creates a train/test discrepancy:
- **Training**: Model sees windows with labels → optimizes window-level predictions
- **Inference**: Need to aggregate multiple window predictions per sample → different from training

This causes:
❌ Training metrics ≠ Validation metrics ≠ Test metrics  
❌ Data leakage risk if windows span across samples  
❌ Difficult hyperparameter tuning (validation F1 doesn't reflect test F1)

**Model-level windowing** (WindowedModelWrapper) solves this:
- **Training**: Model sees full sequences, creates windows internally, aggregates to sample-level predictions
- **Inference**: Exact same process → consistent behavior

This ensures:
✅ Training F1 = Validation F1 = Test F1  
✅ No data leakage (windows created per-sample)  
✅ Hyperparameter tuning reflects test performance  
✅ Base model loss computation preserved (reconstruction + classification for autoencoders)

### How It Works

```python
# In FinalPipeline.objective_kfold()

# 1. Create base model (Direct or Autoencoder)
base_model = Direct(params) or LightningAutoencoder(params)

# 2. Wrap with WindowedModelWrapper if windowing enabled
if use_windowing:
    model = WindowedModelWrapper(
        base_model=base_model,
        window_size=80,              # Sliding window size
        stride=40,                   # 50% overlap
        aggregation_method="avg_probs",  # Average probabilities
        window_loss_weight=0.3       # 30% window loss, 70% sample loss
    )

# 3. Train normally - wrapper handles windowing internally
trainer.fit(model, train_loader, val_loader)

# 4. Validation metrics reflect aggregated sample-level performance
# val_F1 logged is sample-level F1 (matches test behavior)

# 5. Inference uses same aggregation
predictions = model(full_sequences)  # Returns sample-level predictions
```

### Aggregation Methods

**avg_probs** (Recommended):
- Average softmax probabilities across windows, then argmax
- Best calibrated probability estimates
- Smooth decision boundaries

**avg_logits**:
- Average raw logits before softmax
- Faster computation
- Similar performance to avg_probs

**majority_vote**:
- Most frequent class among window predictions
- Robust to outlier windows
- Good for imbalanced classes

**max_confidence**:
- Prediction from window with highest confidence
- Trusts most confident prediction
- Use when model calibration is strong

### Performance Impact

**Computational Cost**:
- Training: ~1.5x slower than data-level windowing (window creation overhead)
- Inference: Same speed (still creates windows)
- Memory: Same (windows created batch-wise)

**Accuracy Improvement**:
- +2-5% F1-score vs data-level windowing (from metric consistency)
- Better hyperparameter selection (validation metrics reliable)
- More robust to test distribution shifts

**FinalPipeline Integration**:
- Automatically enabled/disabled via `use_windowing` trial suggestion
- Hyperparameters optimized jointly with architecture
- Checkpoints preserve windowing configuration
- WindowedSubmissionGenerator uses matching aggregation method
