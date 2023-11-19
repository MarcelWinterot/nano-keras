# Roadmap for the development of nano-keras

## Features left to implement before releasing v1.0.0

### 1. Optimizing the backpropagation of Conv2D using col2im

## Features I  want in v2.0.0

### 1. GPU support

### 2. Batch support

### 3. Optimizers: AdamW, Adamax, Adafactor, Ftrl

### 4. Layers: LSTM, GRU, MHA, Embedding

### 5. Callbacks: LearningRateScheduler, CSVLogger

### 6. Weight initalization strategies: Random uniform, Truncated normal, Zeros, Ones, Xavier uniform, He uniform, Constant, Variance Scalling

## Features I've implemented

### 1. Easily creating and training most simple NN types

### 2. Acitavitons: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax

### 2.1. Loss functions: MAE, MSE, Binary Crossentropy, Categorical Crossentropy, Hinge, Hube

### 3. Optimizers: Adam, SGD, Adagrad, RMSprop, Adadelta, NAdam

### 4. Layers: Input, Dense, Dropout, Flatten, Reshape, MaxPool1D, MaxPool2D, Conv1D, Conv2D

### 5. Regulization techniques: L1, L2, L1L2

### 6. Callbacks: Early stopping

### 7. Weight initialization startegies: Xavier normal, He normal, Random normal

## Possible plans for the future

### Add distribution strategies for training a model on multiple CPUs/GPUs
