# Plans for v. 1.0

## The goal of this project is to learn how things work in popular ML libraries. Of course the perfromance won't be too good as it is written in python but the performance isn't my goal, although I will try to make it as efficient as possible

## Features I want to implement

### 1. Easily creating and training NNs with the library taking care of configuration

### 2. A few activation and loss functions

### 3. A few optimizers to learn how they work

### 4. Custom layer types, like the ones in keras

### 5. Regularization techniques

### 6. Callbacks

## Features I've implemented

### 1. Easily creating and training most NN types

### 2. Acitavitons: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax

### 2.1. Loss functions: MAE, MSE, Binary Crossentropy, Categorical Crossentropy, Hinge, Huber. Although only MSE currently works

### 3. Optimizers: Adam, SGD, Adagrad, RMSprop, Adadelta, NAdam

### 4. Layers: Input, Dense, Dropout, Flatten, Reshape, MaxPool1D, MaxPool2D

### 5. Regulization techniques: L1, L2, L1L2

### 6. Callbacks: Early stopping

### 7. Weight initializations: Xavier, He, Random

## Possible plans for the future

### Rewrite the library in a more efficient language like C++/Rust

### Add GPU support for calculations

### Make the code work in batches

### Add more of everything

### Add distribution strategies for training a model on multiple CPUs/GPUs
