# Pipeline Module

## Description
The Pipeline module provides a unified framework for end-to-end deep learning workflows, orchestrating architecture building, hyperparameter optimization, training, and submission generation. It integrates modular components (DataLoader, Architectures, Optimizer, SubmissionGenerator, Utils) into a cohesive pipeline that enables reproducible experiments through configuration-based architecture construction. The module supports automated hyperparameter tuning with Optuna, flexible architecture selection, and seamless integration with PyTorch Lightning for efficient training workflows.

---

## Main Classes

### `Pipeline`
Central orchestration class that manages the complete machine learning workflow from data loading to model optimization.

**Configuration Parameters:**
- `dataset` (L.LightningDataModule): Training/validation data module
- `test` (L.LightningDataModule): Test data module
- `optimizer` (Optimizer): Hyperparameter optimizer instance
- `dict_config` (dict, optional): Configuration dictionary for architecture and training
- `path_config` (str, optional): Path to YAML configuration file

**Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `dataset: L.LightningDataModule`<br>`test: L.LightningDataModule`<br>`optimizer: Optimizer`<br>`dict_config: dict \| None`<br>`path_config: str \| None` | - | Initialize Pipeline with data modules and configuration. |
| `build_architecture` | `params: dict` | `L.LightningModule` | Build architecture based on configuration parameters. |
| `optimize` | - | `L.LightningModule` | Run hyperparameter optimization and return best model. |

**Supported Architecture Types:**
- `autoencoder_joint`: Joint autoencoder with shared latent space
- `autoencoder_split`: Split autoencoder with separate encoders
- `direct`: Direct classification without reconstruction

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Pipeline                            │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  DataLoader  │───▶│ Architecture │───▶│  Optimizer  │   │
│  │              │    │   Builder    │    │              │   │
│  │ - Train/Val  │    │ - Encoder    │    │ - Optuna     │   │
│  │ - Test Data  │    │ - Decoder    │    │ - TPE        │   │
│  │ - Batching   │    │ - FF Network │    │ - Callbacks  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                    │                    │         │
│         │                    │                    ▼         │
│         │                    │            ┌──────────────┐  │
│         │                    │            │   Training   │  │
│         │                    │            │  (Lightning) │  │
│         │                    │            └──────────────┘  │
│         │                    │                    │         │
│         │                    │                    ▼         │
│         ▼                    ▼            ┌──────────────┐  │
│  ┌──────────────────────────────────────▶│ Submission   │  │
│  │                                        │  Generator   │  │
│  │                                        └──────────────┘  │
│  │                                                          │
│  │         ┌──────────────────────────────────────┐         │
│  └────────▶│  ParameterInterpreter (Utils)        │        │
│            │  - Validation                        │         │
│            │  - Type Checking                     │         │
│            │  - String-to-Object Mapping          │         │
│            └──────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Components

### 1. **DataLoader**
Handles multimodal data loading (time series + global features).

### 2. **Architectures**
Modular building blocks and complete models for classification.

### 3. **Optimizer**
Automated hyperparameter optimization using Optuna.

### 4. **SubmissionGenerator**
Automated CSV submission file creation for competitions.

### 5. **Utils**
Parameter validation and interpretation utilities.
