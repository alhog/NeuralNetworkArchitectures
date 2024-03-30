# NeuralNetworkArchitectures

# Neural Network Architectures Overview

## Table of Contents

1. [Feedforward Neural Networks (FNNs)](#feedforward-neural-networks-fnns)
2. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
3. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
4. [Long Short-Term Memory (LSTM) Networks](#long-short-term-memory-lstm-networks)
5. [Gated Recurrent Unit (GRU) Networks](#gated-recurrent-unit-gru-networks)
6. [Autoencoders](#autoencoders)
7. [Hyperparameter Tuning and Optimization](#hyperparameter-tuning-and-optimization)

## 1. Feedforward Neural Networks (FNNs)

- **Purpose**: FNNs serve as the foundational building blocks of deep learning.
- **Structure**:
    - Input layer, hidden layers, and output layer.
    - Neurons in each layer are fully connected.
- **Use Cases**:
    - Regression, classification, and basic pattern recognition.

## 2. Convolutional Neural Networks (CNNs)

- **Purpose**: Designed specifically for image and spatial data.
- **Structure**:
    - Convolutional layers learn local patterns (edges, textures).
    - Pooling layers downsample feature maps.
    - Fully connected layers handle classification.
- **Use Cases**:
    - Image recognition, object detection, and segmentation.

## 3. Recurrent Neural Networks (RNNs)

- **Purpose**: Handle sequential data (time series, natural language).
- **Structure**:
    - Hidden states capture context.
    - Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) variants address vanishing gradient problem.
- **Use Cases**:
    - Language modeling, speech recognition, and sentiment analysis.

## 4. Long Short-Term Memory (LSTM) Networks

- **Purpose**: Capture long-term dependencies in sequential data.
- **Architecture**:
    - Cell state, gates (reset and update), and hidden state.
- **Use Cases**:
    - Time series prediction, machine translation.

## 5. Gated Recurrent Unit (GRU) Networks

- **Purpose**: Streamlined version of LSTMs.
- **Efficiency**: Achieves comparable performance with faster computation.
- **Key Idea**: Retains memory concept while simplifying architecture.

## 6. Autoencoders

- **Purpose**: Unsupervised learning for dimensionality reduction and feature learning.
- **Architecture**:
    - Encoder maps input to a lower-dimensional representation.
    - Decoder reconstructs input from the latent code.
- **Use Cases**:
    - Anomaly detection, denoising, and feature selection.

## 7. Hyperparameter Tuning and Optimization

- **Importance**: Properly tuned hyperparameters significantly impact model performance.
- **Strategies**:
    - Grid search: Exhaustively search through predefined hyperparameter values.
    - Random search: Randomly sample hyperparameters from a distribution.
    - Bayesian optimization: Model the objective function and choose hyperparameters that maximize/minimize it.

