# Hybrid CNN Model for Pneumonia Detection

This repository contains code for a hybrid Convolutional Neural Network (CNN) model designed for pneumonia detection using chest X-ray images. The model combines the strengths of multiple pre-trained CNN architectures: DenseNet201, InceptionResNetV2, and ResNet50.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This project aims to build a robust model for detecting pneumonia from chest X-ray images. By leveraging transfer learning from multiple state-of-the-art architectures, the hybrid model improves detection accuracy.

## Architecture

The model uses the following base architectures:
- **DenseNet201**
- **InceptionResNetV2**
- **ResNet50**

Each of these architectures is pre-trained on the ImageNet dataset. Their outputs are passed through additional convolutional layers and GlobalAveragePooling2D layers before being concatenated. The concatenated features are then passed through fully connected layers to produce the final classification output.

## Dataset

The dataset used in this project includes labeled chest X-ray images:

- **Training Data**: Located in the `chest_xray/test` directory.
- **Validation Data**: Located in the `validation1/val` directory.

### Directory Structure

```
- chest_xray/
    - test/
        - NORMAL/
        - PNEUMONIA/
- validation1/
    - val/
        - NORMAL/
        - PNEUMONIA/
```

## Installation

### Prerequisites

Ensure you have the following libraries installed:

- Python 3.10+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- SciPy

You can install the required libraries using pip:

```bash
pip install tensorflow numpy pandas matplotlib scipy
```

## Usage

### Model Definition

The model architecture is defined in the `hybrid_cnn_model` function. It integrates the DenseNet201, InceptionResNetV2, and ResNet50 models, applying additional convolutional layers and global average pooling to each.

### Learning Rate Scheduler

A custom learning rate scheduler is implemented in the `lr_scheduler` function to adjust the learning rate during training.

### Data Generators

- **Training Data Generator**: Applies data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping.
- **Validation Data Generator**: Only rescales the images without any augmentation.

## Training

To train the model, run the following command:

```python
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[lr_scheduler_callback]
)
```

Training will proceed for 100 epochs, with a learning rate scheduler adjusting the learning rate at predefined intervals.

## Evaluation

The model's performance can be evaluated using the validation dataset, as shown in the training script. Validation accuracy and loss are printed after each epoch.

### Example Output

```
Epoch 1/100
39/39 [==============================] - 18s 467ms/step - loss: 0.1953 - accuracy: 0.9279 - val_loss: 2.2249 - val_accuracy: 0.8412
...
```

## Results

- The model achieved a high validation accuracy, consistently reaching over 98% after tuning the learning rate.

## Reference

For more detailed information on this project, please refer to the published paper:  
**"Hybrid CNN Model for Pneumonia Detection"**  
Available on [IEEE Xplore](https://ieeexplore.ieee.org/document/10425735).
