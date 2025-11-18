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
<pre>
├── models/                   # Model architectures
│   ├── lwn.py                # Base lightweight network
├── Tools/                    # Utility functions and tools
│   ├── Grad-CAM.py           # Comparison of heatmaps across multiple networks
│   ├── Robustness.py         # Robustness testing in the paper
│   └── confusion_matrix.py   # Confusion matrix of three view and four view
├── SSLTRAIN-AS.py            # Semi-supervised training for Aortic Stenosis
├── SSLTRAIN-VIEW.py          # Semi-supervised training for View classification
├── TRAIN-AS.py               # Supervised training for Aortic Stenosis
├── TRAIN-VIEW.py             # Supervised training for View classification
├── augment.py                # Data augmentation utilities
├── split.py                  # Data splitting utilities
└── requirements.txt          # Python dependencies
</pre>
### Process the dataset

The dataset required your application to be used (https://tmed.cs.tufts.edu/tmed_v2.html)
# Running experiments
Split dataset:
You can divide the dataset according to the TMED official classification in this way:
```bash
python split.py --data_dir /path/to/data --output_dir ./splits 
```
Training:
You can train the model in fully supervised and semi supervised ways through TRAIN-VIEW.py 
SSLTRAIN-AS.py 
SSLTRAIN-VIEW.py
TRAIN-AS.py
TRAIN-VIEW.py  




