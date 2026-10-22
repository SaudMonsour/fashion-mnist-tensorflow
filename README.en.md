# Fashion-MNIST Classification with TensorFlow

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comparative study evaluating Convolutional Neural Networks (CNN) against Multi-Layer Perceptrons (MLP) for recognizing ten clothing categories from grayscale images under a fixed 5-epoch training budget.

---

## Overview

* **Task:** Multi-class image classification (10 categories).
* **Dataset:** 70,000 total samples (60,000 training, 10,000 official test) at 28 × 28 × 1 resolution.
* **Top Model:** **TensorFlow CNN** (Holdout Macro-F1 = **0.8871**).
* **Baseline Comparison:** Dummy Classifier (Holdout Macro-F1 = 0.0182).
* **Environment:** Python 3.12 with TensorFlow 2.20.

---

## Key Results

### Training Cross-Validation (3-Fold Stratified)

| Model | Macro-F1 | Fold SD | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **CNN** | **0.8822** | 0.0013 | 105,866 |
| Dense MLP | 0.8670 | 0.0021 | 109,386 |
| Dummy Baseline | 0.0182 | 0.0000 | 0 |

* **Protocol:** 3-fold cross-validation on the 60,000 training dataset (40,000 fit / 20,000 validation per fold).
* **Hyperparameters:** Fixed 5 epochs, batch size of 256, Adam optimizer ($\text{lr}=0.001$), and dropout rate of 0.25. Pixels rescaled to $[0, 1]$.
* **Architecture:** 
  * **CNN:** Two $3 \times 3$ Conv layers (16 and 32 filters) with Max Pooling, followed by a 64-unit Dense layer.
  * **MLP:** Two Dense layers (128 and 64 units).

### Final Holdout Evaluation (Official Test Set - 10,000 Samples)

| Model | Macro-F1 | Accuracy | Balanced Accuracy |
| :--- | :--- | :--- | :--- |
| **CNN** | **0.8871** | **0.8874** | **0.8874** |
| Dummy Baseline | 0.0182 | 0.1000 | 0.1000 |

* **Refitting:** The selected CNN was refitted on all 60,000 training images for 5 epochs before test evaluation.
* **Error Analysis:** 1,126 out of 10,000 test images were misclassified. The *Shirt* class exhibited the highest error rate (F1 = 0.6937), frequently misclassified as *T-shirt/top* (136 cases) or *Coat* (95 cases).

---

## Visualizations

| Model Comparison | Learning Curves |
| :---: | :---: |
| ![Training cross-validation Macro-F1 and fold SD](figures/model-comparison.png) | ![Mean training and validation accuracy and loss](figures/learning-curves.png) |

| Confusion Matrix | Error Examples |
| :---: | :---: |
| ![CNN confusion matrix on the official test set](figures/confusion-matrix.png) | ![First twelve test errors in source order](figures/error-examples.png) |

| Dataset Examples | Data Quality |
| :---: | :---: |
| ![First training image from each class](figures/dataset-examples.png) | ![Training class counts and data audit](figures/data-quality.png) |

---

## Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-username/fashion-mnist-tensorflow.git](https://github.com/your-username/fashion-mnist-tensorflow.git)
cd fashion-mnist-tensorflow
pip install -r requirements.txt
