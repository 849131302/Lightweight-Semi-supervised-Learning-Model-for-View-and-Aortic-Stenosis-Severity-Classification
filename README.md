# Lightweight Semi-supervised Learning Model for View and Aortic Stenosis Severity Classification
A lightweight semi-supervised learning framework for medical image classification tasks, specifically designed for echocardiographic view classification and aortic stenosis severity assessment.

## 📖 Overview

This project implements semi-supervised learning models that can effectively utilize both labeled and unlabeled medical image data to improve classification performance for two important cardiac imaging tasks:

- **View Classification**: Identifying the specific echocardiographic view
- **Aortic Stenosis Severity Classification**: Assessing the severity of aortic stenosis from echocardiograms


## 🚀 Features

- **Lightweight Models**: Efficient architectures suitable for medical imaging applications
- **Semi-supervised Learning**: Leverages both labeled and unlabeled data using MixMatch and other SSL techniques
- **Dual Task Support**: Handles both view classification and aortic stenosis assessment
- **Data Augmentation**: Comprehensive augmentation pipeline for medical images
- **Flexible Training**: Both supervised and semi-supervised training options
- **Reproducible Results**: Complete experimental setup for reproducible research

## 📁 Project Structure
├── models/ # Model architectures
│ ├── base_model.py # Base model class
│ ├── resnet.py # ResNet implementations
│ └── ssl_models.py # Semi-supervised models
├── Tools/ # Utility functions and tools
│ ├── metrics.py # Evaluation metrics
│ └── visualization.py # Visualization utilities
├── SSLTRAIN-AS.py # Semi-supervised training for Aortic Stenosis
├── SSLTRAIN-VIEW.py # Semi-supervised training for View classification
├── TRAIN-AS.py # Supervised training for Aortic Stenosis
├── TRAIN-VIEW.py # Supervised training for View classification
├── augment.py # Data augmentation utilities
├── split.py # Data splitting utilities
└── requirements.txt # Python dependencies

### Process the dataset

https://tmed.cs.tufts.edu/tmed_v2.html
# Running experiments
python TRAIN-VIEW.py



