# Lightweight-Semi-supervised-Learning-Model-for-View-and-Aortic-Stenosis-Severity-Classification
Lightweight Semi-supervised Learning Model for View and Aortic Stenosis Severity Classification
A lightweight semi-supervised learning framework for medical image classification tasks, specifically designed for echocardiographic view classification and aortic stenosis severity assessment.

Overview
This project implements semi-supervised learning models that can effectively utilize both labeled and unlabeled medical image data to improve classification performance for two important cardiac imaging tasks:

View Classification: Identifying the specific echocardiographic view

Aortic Stenosis Severity Classification: Assessing the severity of aortic stenosis from echocardiograms

Project Structure
text
├── models/              # Model architectures
├── Tools/               # Utility functions and tools
├── idea/                # IDE configuration files
├── SSLTRAIN-AS.py       # Semi-supervised training for Aortic Stenosis
├── SSLTRAIN-VIEW.py     # Semi-supervised training for View classification
├── TRAIN-AS.py          # Supervised training for Aortic Stenosis
├── TRAIN-VIEW.py        # Supervised training for View classification
├── augment.py           # Data augmentation utilities
├── split.py             # Data splitting utilities
└── README.md            # Project documentation
Features

Lightweight Models: Efficient architectures suitable for medical imaging applications

Semi-supervised Learning: Leverages both labeled and unlabeled data

Dual Task Support: Handles both view classification and aortic stenosis assessment

Data Augmentation: Comprehensive augmentation pipeline for medical images

Flexible Training: Both supervised and semi-supervised training options

Installation
Clone the repository:

bash
git clone https://github.com/your-username/Lightweight-Semi-supervised-Learning-Model-for-View-and-Aortic-Stenosis-Severity-Classification.git
cd Lightweight-Semi-supervised-Learning-Model-for-View-and-Aortic-Stenosis-Severity-Classification
Install required dependencies:

bash
pip install -r requirements.txt
Usage
Data Preparation
bash
python split.py          # Split your dataset
python augment.py        # Apply data augmentation
Training
Supervised Training:

bash
python TRAIN-VIEW.py    # Train view classification model
python TRAIN-AS.py      # Train aortic stenosis model
Semi-supervised Training:

bash
python SSLTRAIN-VIEW.py # Semi-supervised view classification
python SSLTRAIN-AS.py   # Semi-supervised aortic stenosis
Model Architecture
The project implements lightweight convolutional neural networks optimized for:

Computational efficiency

Medical image characteristics

Semi-supervised learning paradigms

Transfer learning capabilities


Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.

Contact
For questions or collaborations, please contact [zhangyifeng@nefu.edu.cn] or open an issue in this repository.
