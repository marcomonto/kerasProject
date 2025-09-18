# KeraProject - Neural Networks Experiments with Keras

Machine learning project implementing neural network experiments using Keras/TensorFlow for three different ML problems: breast cancer binary classification, car price regression, and sonar signal classification.

## 📋 Project Overview

This project includes three comprehensive machine learning analyses:

1. **Breast Cancer Classification**: Binary classification to distinguish malignant from benign tumors
2. **Cars Regression**: Regression to predict car prices
3. **Sonar Classification**: Binary classification to distinguish metal from rock using sonar signals

## 🗂️ Project Structure

```
kerasProject/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── ML_Theory_Summary.md               # Neural networks theory summary
├── ML2025_lab5_NeuralNets.pdf        # Lab documentation
│
├── breast_cancer_analysis.py          # Main script - Breast cancer
├── breastCancerModules/               # Breast cancer analysis modules
│   ├── __init__.py
│   ├── data_handler.py               # Data management and preprocessing
│   ├── models.py                     # Model architectures
│   ├── training.py                   # Training and experiments
│   └── visualization.py              # Plots and visualizations
│
├── cars_regression_analysis.py       # Main script - Car regression
├── carsModules/                       # Car regression modules
│   ├── data_handler.py
│   ├── model.py
│   ├── training.py
│   └── visualization.py
│
├── sonar_classification_analysis.py  # Main script - Sonar classification
├── sonarModules/                      # Sonar classification modules
│   ├── __init__.py
│   ├── data_handler.py
│   ├── model.py
│   ├── training.py
│   └── visualization.py
│
├── results/                          # Experiment results
└── venv/                            # Virtual environment
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone repository
git clone <repository-url>
cd kerasProject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies
- TensorFlow 2.20.0
- Scikit-learn 1.7.1
- NumPy 2.2.6
- Pandas 2.2.3
- Matplotlib 3.10.6

## 📊 Implemented Experiments

### 1. Breast Cancer Classification
**Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
**Problem**: Binary classification (Malignant/Benign)

**Features**:
- Architecture comparison: Basic (64→32→1) vs Funnel
- Experiments on epochs, batch_size, and architecture
- Metrics: accuracy, precision, recall, F1-score
- Complete visualizations with confusion matrix

**Execution**:
```bash
python breast_cancer_analysis.py
```

### 2. Cars Regression
**Dataset**: Car price prediction based on customer characteristics
**Problem**: Regression (continuous value prediction)

**Features**:
- Age, Gender, Miles/day, Personal debt, Monthly income
- Architecture: 64→32→1 with linear activation
- Loss function: MSE (Mean Squared Error)
- Feature importance analysis

**Execution**:
```bash
python cars_regression_analysis.py
```

### 3. Sonar Classification
**Dataset**: 208 samples, 60 sonar features
**Problem**: Binary classification (Metal/Rock)

**Features**:
- Base model and improved model with Dropout
- Early stopping to prevent overfitting
- Parameters: epochs=100, batch_size=5
- Prediction confidence analysis

**Execution**:
```bash
python sonar_classification_analysis.py
```

## Architectures and Configurations

### Activation Functions
- **ReLU**: Hidden layers (computational efficiency)
- **Sigmoid**: Binary classification output (0-1)
- **Linear**: Regression output (continuous values)

### Optimizers
- **Adam**: Used in all experiments for adaptive convergence

### Loss Functions
- **Binary Crossentropy**: Binary classification
- **MSE**: Regression

### Regularization Techniques
- **Validation Split**: 80/20 for overfitting monitoring
- **Early Stopping**: Automatic halt when validation loss stops improving
- **Dropout**: In improved sonar model

## 📈 Results and Visualizations

Each experiment automatically generates:

1. **Training History**: Accuracy/loss plots over epochs
2. **Classification Analysis**: Confusion matrix, per-class metrics
3. **Prediction Analysis**: Confidence distribution, prediction examples
4. **Comprehensive Reports**: Complete performance analysis

Results are saved in the `results/grafici/` folder.
