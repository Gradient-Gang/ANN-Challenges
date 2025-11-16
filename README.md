# Gr4dient G4ng - ANN Challenges

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.5%2B-purple.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a comprehensive implementation of artificial neural network models for the ANN Challenges course. The project provides a modular, scalable pipeline for deep learning workflows, featuring automated hyperparameter optimization, multimodal data processing, and flexible architecture design using PyTorch Lightning.

---

## 🎯 Key Features

- **🔧 Modular Pipeline Architecture**: End-to-end machine learning workflow with configurable components
- **🧠 Flexible Neural Network Architectures**: Autoencoders, direct classifiers, and custom modular building blocks
- **📊 Multimodal Data Processing**: Simultaneous handling of time series and tabular data
- **🔍 Automated Hyperparameter Optimization**: Optuna-based TPE optimization with early stopping
- **⚡ Lightning Integration**: PyTorch Lightning for efficient training, distributed computing, and GPU support
- **📝 Configuration-Driven Design**: YAML-based configuration for reproducible experiments
- **🎨 Data Preprocessing Pipeline**: Feature engineering, normalization, PCA, and visualization tools
- **📤 Competition Submission Generator**: Automated CSV generation for Kaggle-style competitions

---

## 📂 Project Structure

```
ANN-Challenges/
├── src/
│   └── GradientGang/
│       ├── Pipeline/
│       │   ├── Pipeline.py                         # FinalPipeline orchestration class
│       │   ├── Architectures/
│       │   │   ├── LightningAutoencoder.py        # Autoencoder with reconstruction
│       │   │   ├── Direct.py                       # Direct classification model
│       │   │   ├── WindowedModelWrapper.py        # Model-level windowing wrapper
│       │   │   ├── Encoder.py                      # Modular encoder blocks
│       │   │   ├── Decoder.py                      # Modular decoder blocks
│       │   │   └── FeedForward.py                 # Feedforward networks
│       │   ├── DataLoader/
│       │   │   └── DataLoader.py                  # Multimodal data loading + K-fold CV
│       │   ├── Optimizer/
│       │   │   ├── Optimizer.py                   # Base optimizer interface
│       │   │   └── OptunaOptimizer.py             # Optuna TPE optimization
│       │   ├── SubmissionGenerator/
│       │   │   ├── SubmissionGenerator.py         # Standard CSV submission
│       │   │   └── WindowedSubmissionGenerator.py # Windowed prediction aggregation
│       │   └── Utils/
│       │       ├── ParameterInterpreter.py        # Parameter validation
│       │       ├── FeatureSelector.py             # Supervised feature selection
│       │       └── EnsembleModels.py              # Model ensemble utilities
│       └── PreProcessing/
│           ├── PreProcessor.py                    # Data preprocessing pipeline
│           └── DataExploration/
│               ├── DataCleaning.ipynb
│               ├── DataVisualization.ipynb
│               └── RandomAnalysis.ipynb
├── dataset/
│   ├── Pirate/                                    # Original datasets
│   └── PirateProcessed/                           # Preprocessed datasets
│       ├── pirate_pain_train.csv                  # Processed time series
│       ├── pirate_pain_test.csv
│       ├── pirate_pain_train_labels.csv           # Training labels
│       ├── train_global_features.csv              # Global features
│       ├── test_global_features.csv
│       ├── class_weights.yaml                     # Computed class weights
│       └── selected_features.txt                  # Selected feature names
├── Notebook/
│   ├── FinalPipelineTest.ipynb                    # Pipeline testing
│   └──  preprocessing.ipynb                        # Preprocessing workflows
├── tests/                                         # Comprehensive test suite (293 tests)
├── Deliverables/
│   ├── Report1/                                   # LaTeX project report
│   ├── Tracker/
│   │   └── ideas_tracker.md                       # Project ideas and notes
│   └── UML/
│       ├── architecture.puml                      # PlantUML architecture diagram
│       ├── UML.drawio                             # DrawIO diagram
│       └── UML_drawio.png                         # Architecture visualization
├── Submissions/                                   # Generated submission files
└── pyproject.toml                                 # Project dependencies & metadata
```

---

## 🏗️ Architecture Overview

![Architecture Diagram](Deliverables/UML/UML_drawio.png)

The system follows a modular design with clear separation of concerns:

### PreProcessor Module
Comprehensive data preprocessing pipeline for feature engineering, normalization, PCA dimensionality reduction, and data visualization. Handles time series transformations, class weight computation, and feature selection.

📖 [PreProcessing Documentation](src/GradientGang/PreProcessing/README.md)

### Pipeline Module
Central orchestration component managing the complete machine learning workflow. Dynamically constructs models from configuration, handles data management, integrates hyperparameter optimization, and leverages PyTorch Lightning for efficient training.

📖 [Pipeline Documentation](src/GradientGang/Pipeline/README.md)

#### Architecture Module
Flexible building blocks for neural network construction including Encoders (Conv1D/2D, LSTM, GRU, RNN, MultiScaleCNN), Decoders (transpose convolutions, recurrent layers), FeedForward networks, LightningAutoencoder (joint/split latent spaces), Direct classification models, and **WindowedModelWrapper** for model-level windowing with aggregation strategies.

📖 [Architectures Documentation](src/GradientGang/Pipeline/Architectures/README.md)

#### DataLoader Module
Multimodal data processing for time series and tabular features. Implements PyTorch Dataset for combined modalities and Lightning DataModule with automatic train/validation splits, batching, and efficient data loading.

📖 [DataLoader Documentation](src/GradientGang/Pipeline/DataLoader/README.md)

#### Optimizer Module
Automated hyperparameter tuning using Optuna with TPE sampler for efficient Bayesian optimization. Features early stopping, model checkpointing, and reproducible results with fixed seeds.

📖 [Optimizer Documentation](src/GradientGang/Pipeline/Optimizer/README.md)

#### Submission Module
Automated CSV generation for competition submissions. Maps model predictions to labels, handles batch processing, windowed prediction aggregation (avg_probs/majority_vote/max_confidence), and creates properly formatted submission files for Kaggle-style competitions.

📖 [SubmissionGenerator Documentation](src/GradientGang/Pipeline/SubmissionGenerator/README.md)

#### Utils Module
Parameter validation and interpretation utilities. Ensures configuration correctness, validates required parameters, and provides mapping between architecture types and their implementations.

📖 [Utils Documentation](src/GradientGang/Pipeline/Utils/README.md)


---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Gradient-Gang/ANN-Challenges.git
cd ANN-Challenges

# Install dependencies using Poetry (recommended)
poetry install

# Activate virtual environment
poetry shell

# Or install with pip
pip install -e .
```

### Running Experiments

```python
from GradientGang.Pipeline.Pipeline import FinalPipeline

# Configure pipeline
params = {
    "project_name": "PiratePain",
    "study_name": "experiment_v1",
    "database_url": "postgresql://user:pass@localhost:5432/optuna",
    "data_params": {
        "train_path": "dataset/PirateProcessed/pirate_pain_train.csv",
        "test_path": "dataset/PirateProcessed/pirate_pain_test.csv",
        "train_labels_path": "dataset/PirateProcessed/pirate_pain_train_labels.csv",
        "train_global_path": "dataset/PirateProcessed/train_global_features.csv",
        "test_global_path": "dataset/PirateProcessed/test_global_features.csv",
        "class_weights_path": "dataset/PirateProcessed/class_weights.yaml",
        "n_folds": 5,
        "batch_size": 32,
        "num_workers": 4
    },
    "submission_path": "./submissions"
}

# Create pipeline
pipeline = FinalPipeline(params)

# Run optimization
pipeline.optuna_optimize(n_trials=100)

# View results
pipeline.study_summary()

# Generate submission
submission_df = pipeline.create_submission()
```

---

## 🧪 Testing

[![Tests](https://img.shields.io/badge/Tests-293%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen.svg)](tests/)

The project includes a comprehensive test suite with **293 tests** achieving **90% code coverage**.

### Run Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage report
poetry run pytest --cov=src --cov-report=term-missing tests/
```

For detailed test documentation, see [tests/README.md](tests/README.md).

---

## 📚 Documentation

Each module includes comprehensive README files:
- [Pipeline Documentation](src/GradientGang/Pipeline/README.md)
- [Architectures Documentation](src/GradientGang/Pipeline/Architectures/README.md)
- [DataLoader Documentation](src/GradientGang/Pipeline/DataLoader/README.md)
- [Optimizer Documentation](src/GradientGang/Pipeline/Optimizer/README.md)
- [SubmissionGenerator Documentation](src/GradientGang/Pipeline/SubmissionGenerator/README.md)
- [Utils Documentation](src/GradientGang/Pipeline/Utils/README.md)
- [PreProcessing Documentation](src/GradientGang/PreProcessing/README.md)

---


## 👥 Team

**Gr4dient G4ng Team:**
- **Paolo Ginefra** - [ginefra.paolo@gmail.com](mailto:ginefra.paolo@gmail.com)
- **Martina Missana** - [martina.missana@mail.polimi.it](mailto:martina.missana@mail.polimi.it)
- **Lorenzo Pettenuzzo** - [lorenzo.pettenuzzo@mail.polimi.it](mailto:lorenzo.pettenuzzo@mail.polimi.it)
- **Matteo Pasqual** - [matteoromilio.pasqual@mail.polimi.it](mailto:matteoromilio.pasqual@mail.polimi.it)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Repository**: [Gradient-Gang/ANN-Challenges](https://github.com/Gradient-Gang/ANN-Challenges)
- **Course**: Artificial Neural Networks and Deep Learning
- **Institution**: Politecnico di Milano

---

## 🙏 Acknowledgments

This project was developed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano. Special thanks to the course instructors for their guidance and support.