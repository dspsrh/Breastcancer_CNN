# Breast Cancer Detection using CNN

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A depth-wise Convolutional Neural Network (CNN) implementation using `SeparableConv2D` to classify 50x50 histopathology image patches as **benign** (negative) or **malignant** (positive).

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)
- [Notes & Caveats](#notes--caveats)

## 🔍 Overview
This project provides a simple, end-to-end workflow for binary breast cancer classification:
1.  **Data Preparation**: Scans the IDC dataset, balances the classes, and serializes the data into pickle files for fast loading.
2.  **Training**: Trains a lightweight CNN optimized for small image patches using separable convolutions.
3.  **Inference**: Loads the saved model to perform predictions on new images and evaluating performance.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `dataset.py` | Pre-processing script. Reads images, balances the dataset, and saves `XTrain/XTest` and `yTrain/yTest` pickles. |
| `Network.py` | Model definition and training script. Saves the best model to `CancerNet.model`. |
| `detectCancer.py` | Inference script. Loads the model and runs predictions/evaluation. |
| `test.py` | scratchpad script for quickly loading and checking pickle data. |
| `CancerNet.model/` | (Generated) Saved model artifact. |
| `*.pickle` | (Generated) Serialized dataset files. |

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10** installed. It is recommended to use a virtual environment.

```bash
conda create -n bc_cnn python=3.10 -y
conda activate bc_cnn
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python tensorflow keras
```

### 2. Dataset Setup
Download the **IDC dataset** (e.g., `IDC_regular_ps50_idx5`) and place it locally.

> [!IMPORTANT]
> You **must** update the path in `dataset.py` to point to your local dataset location before running.
> ```python
> # Example in dataset.py
> imagePatches = glob('/path/to/YOUR_DATASET/IDC_regular_ps50_idx5/**/*.png', recursive=True)
> ```

### 3. Run the Workflow

**Step 1: Process Data**
Sample the dataset and create pickle files:
```bash
python dataset.py
```
*Output: `XTrain.pickle`, `XTest.pickle`, `yTrain.pickle`, `yTest.pickle`*

**Step 2: Train Model**
Train the CNN. This will typically run for 50 epochs.
```bash
python Network.py
```
*Output: Saves `CancerNet.model` and displays training plots.*

**Step 3: Evaluation**
Run predictions and view the classification report.
```bash
python detectCancer.py
```

## 🧠 Model Architecture
The model uses **Separable Convolutions** to reduce parameters while maintaining accuracy, making it efficient.
- **Input**: 50x50x3 RGB images.
- **Layers**: Stacks of `SeparableConv2D` + `BatchNormalization` + `MaxPooling` + `Dropout`.
- **Classifier**: Flatten layer feeding into a Dense(256) layer and a final Softmax output.
- **Optimizer**: Adagrad (LR=0.01).
- **Loss Function**: Binary Crossentropy.

## ⚠️ Notes & Caveats
- **Memory Usage**: `dataset.py` loads the sampled dataset into RAM. Validating on the full IDC dataset might require significant memory. Consider modifying it to use `tf.data` or generators for larger scale training.
- **Paths**: The code currently uses hardcoded paths in some places. Please check `dataset.py` ensuring the glob pattern matches your OS (Windows vs Linux).
- **Labels**: The project uses one-hot encoded labels (`[0, 1]` vs `[1, 0]`) with `binary_crossentropy`.
