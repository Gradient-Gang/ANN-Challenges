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
│       │   ├── Pipeline.py                    # Main orchestration class
│       │   ├── Architectures/
│       │   │   ├── LightningAutoencoder.py   # Autoencoder with reconstruction
│       │   │   ├── Direct.py                  # Direct classification model
│       │   │   ├── Encoder.py                 # Modular encoder blocks
│       │   │   ├── Decoder.py                 # Modular decoder blocks
│       │   │   └── FeedForward.py            # Feedforward networks
│       │   ├── DataLoader/
│       │   │   └── DataLoader.py             # Multimodal data loading
│       │   ├── Optimizer/
│       │   │   ├── Optimizer.py              # Base optimizer interface
│       │   │   └── OptunaOptimizer.py        # Optuna-based optimization
│       │   ├── SubmissionGenerator/
│       │   │   └── SubmissionGenerator.py    # CSV submission generation
│       │   └── Utils/
│       │       └── ParameterInterpreter.py   # Parameter validation
│       └── PreProcessing/
│           ├── PreProcessor.py               # Data preprocessing pipeline
│           └── DataExploration/
│               ├── DataCleaning.ipynb
│               ├── DataVisualization.ipynb
│               └── RandomAnalysis.ipynb
├── dataset/
│   ├── Pirate/                               # Original datasets
│   └── PirateProcessed/                      # Preprocessed datasets
├── Notebook/
│   ├── Architectures.ipynb                   # Architecture experiments
│   ├── DataLoading.ipynb                     # Data loading examples
│   ├── preprocessing.ipynb                   # Preprocessing workflows
│   └── Params/
│       └── preprocessing_params.yaml         # Preprocessing configuration
├── tests/                                    # Comprehensive test suite
├── Deliverables/
│   ├── Report1/                              # LaTeX project report
│   ├── Tracker/
│   │   └── ideas_tracker.md                  # Project ideas and notes
│   └── UML/
│       ├── architecture.puml                 # PlantUML architecture diagram
│       ├── UML.drawio                        # DrawIO diagram
│       └── UML_drawio.png                    # Architecture visualization
└── pyproject.toml                            # Project dependencies & metadata
```

---

## 🏗️ Architecture Overview

![Architecture Diagram](Deliverables/UML/UML_drawio.png)

The system follows a modular design with clear separation of concerns:

### Pipeline Module
The central orchestration component that manages the complete machine learning workflow:
- **Architecture Building**: Dynamically constructs models from configuration
- **Data Management**: Handles train/validation/test splits and batching
- **Optimization**: Integrates with Optuna for hyperparameter search
- **Training**: Leverages PyTorch Lightning for efficient training loops

### Architecture Modules
Flexible building blocks for neural network construction:
- **Encoder**: Supports Conv1D/2D, LSTM, GRU, RNN, and Linear layers
- **Decoder**: Transpose convolutions and recurrent layers for reconstruction
- **FeedForward**: Configurable fully connected networks with normalization
- **LightningAutoencoder**: Complete autoencoder with joint/split latent spaces
- **Direct**: Direct classification without reconstruction overhead

### Data Pipeline
Multimodal data processing for time series and tabular features:
- **TimeSeriesAndGlobalDataset**: PyTorch Dataset for combined modalities
- **DataModule**: Lightning DataModule with automatic train/val splits
- **PreProcessor**: Comprehensive preprocessing with PCA and feature selection

### Optimization
Automated hyperparameter tuning with Optuna:
- **TPE Sampler**: Efficient Bayesian optimization
- **Early Stopping**: Prevents overfitting during optimization
- **Model Checkpointing**: Saves best models automatically
- **Reproducible**: Fixed seeds for deterministic results

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