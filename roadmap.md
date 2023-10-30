# Plans for v. 1.0

### The goal of this project is to learn how things work in popular ML libraries. Of course the perfromance won't be too good as it is written in python but the performance isn't my goal, although I will try to make it as efficient as possible.

## Features I want to implement

### 1. Freely creating and training NNs with the library taking care of configuration

### 2. All activation and loss functions

### 3. A few optimizers to learn how they work

### 4. Custom layer types, like the ones in keras

### 5. Regularization techniques

### 6. Callbacks

## Features I've implemented

### 1. Freely creating and training NNs with the library taking care of configuration

### 2. A few activation and loss function

### 3. Optimizers: Adam, SGD

### 4. Layers: Dense, Dropout, Flatten, Reshape

### 5. L1, L2 and L1L2 regulization techniques

### 6. Callbacks - Early stopping

## Features I'm currently working on

### 1. Adding a few optimizers

#### I'm planning on adding: RMSprop, Adagrad, Adadelta, Nadam

### 2. Adding layer types

#### Not implemented layers I'm planning to add in v. 1.0: Convolutional, Pooling

#### If I have time I'll also implement: Concantanete layer, LSTM or GRU

### 3. Fixing the derivatives of loss functions

#### Only the MSE works currently and I want each loss function to work

#### It will take a while as I'm currently learning how to calculate derivatives and I want to calculate them myself

# Possible plans for v. 2.0

### Rewrite the library in Rust

### Make the code work in batches
