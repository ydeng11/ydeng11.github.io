---
title: Handmade Neural Network with Numpy
date: 2021-01-02 17:00:08
tags: 
- neural network
- coding
category: neural network
mathjax: True
index_img: https://raw.githubusercontent.com/ydeng11/typora_pics/master/typora20200603213936-79723.jpeg
---

![c75f3cc2-f37d-4b2e-a41b-1db5a335677c](https://raw.githubusercontent.com/ydeng11/typora_pics/master/typora20200603213936-79723.jpeg) This is the flow of neural network including feed forward and backpropagation. Here we used the structures in the code as an exmaple to show how the dimensions change in each layer to better understand the structures. It is obvious we can iterate the layers backward to pass derivatives and  update the gradients.
In addition, the top row shows the backpropagtion and the division sympol represents the partial derivatives. We can easily get the partial derivatives in each layer via: $\frac{\partial U_{next}}{\partial W_{curr}}$,  $\frac{\partial U_{next}}{\partial U_{curr}}$,  : $\frac{\partial U_{next}}{\partial b_{curr}}$

```python
 
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn import datasets
```

### Design the network structure

- Each layer contains the weights/bias and activation union


```python
structures = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]
```

### Initiate the parameters

- The weights can be random number and bias are preferred to be small postive values in order to pass the relu in the beginning.


```python
def init_layers(structures, seed = 1105):
    params = {}
    for i, structure in enumerate(structures):
        params["W_{}".format(i)] = np.random.randn(structure["input_dim"], structure["output_dim"])/10
        params["b_{}".format(i)] = np.random.randint(1,10, (1, structure["output_dim"]))/100
    return params
```

### The forward and backword activation union

- During back propagation, it is appraent we would need use the output value before activation in feed forward process. We would need to save the ouput before and after activation in each layer for back propagation later.


```python
def relu(U):
    U[U < 0] = 0
    return U
 
def sigmoid(U):
    return np.divide(1, (1+np.exp(-1*U)))
 
def relu_backward(du, U):
    du[U < 0] = 0
    return du
 
def sigmoid_backward(du, U):
    sig = sigmoid(U) * (1 - sigmoid(U))
    return du*sig
```

So, we return two values in single_layer_feedforward function corresponding to the activated output and output which doesn't. The activated output will be feed as input into the next layer and the unactivated output will be used in backpropagation - the reason is we need the partial derivatives of activation union to its input.


```python
def single_layer_feedforward(A, W, b, activation_func):
    return activation_func(A@W + b), A@W + b
```

- Duing feed forward process, we start with features (X), and go through each layer till the final output.


```python
def feedforward(X, structures, params):
    U_curr = X
    for i, structure in enumerate(structures):
        # set_trace()
        W_curr = params["W_" + str(i)]
        b_curr = params["b_" + str(i)]
        params["U_input_" + str(i)] = U_curr
        if structure["activation"] == "relu":
            activation_func = relu
        elif structure["activation"] == "sigmoid":
            activation_func = sigmoid
        else:
            print("no supported activation")
            exit
        U_next, U_curr = single_layer_feedforward(U_curr, W_curr, b_curr, activation_func)
        params["U_post_activation_" + str(i)] = U_next
        params["U_prior_activation_" + str(i)] = U_curr
        U_curr = U_next
    return U_curr
```

### Loss function

Here we used the negative log-loss as the $\mathbb{L}$ (total loss) to minimize.


```python
def negativelogloss(output, y):
    # set_trace()
    return np.squeeze(-1 * sum(y * np.log10(output) + (1 - y)*np.log10(1-output)) / len(y))
 
def get_accuracy(y_true, y_predicted):
    predicted_class = y_predicted.copy()
    predicted_class.reshape(-1)
    predicted_class[predicted_class > 0.5] = 1
    predicted_class[predicted_class <= 0.5] = 0
    return accuracy_score(y_true, predicted_class)
```

### Backpropagtion process

- During backpropagtion, we passed the partial derivatives based on the chain rules.
- In the single layer backward, it is obvious we will feed the previous derivatives into the layer in bottom-up order, then the derivatives will multiply the partial derivatives in the layer to generate the accumulative derivatives for next layer.
  During this process, we will also save the gradient of weights and bias (average gradients) in table for update later.


```python
def single_layer_backward(dz, U_input, U_prior_activation, W, activation_func):
    m = len(dz)
    dz = activation_func(dz, U_prior_activation)
    gradient_W = (U_input.T @ dz) / m
    gradient_b = np.mean(dz, axis = 0)
    dz = dz @ W.T
#     set_trace()
    return dz, gradient_W, gradient_b
```

- During the whole backpropagtion process, we started from the partial derivatives of loss function to the y_hat (output of feed forward).
- In each layers, we will accumulate the derivatives and calculate the gradient of W and b.
- The accumulated derivatives will be passed to next layer.
- The gradient of W and b in each layer will stored in grads table and we can update them later.

 

```python
grads_table = {}
def backward(output, y, structures, params):
    dz_prev = -(np.divide(y, output) - np.divide(1-y, 1 - output))
    # set_trace()
    i = len(structures) - 1
    while i >= 0:
        W_curr = params["W_" + str(i)]
        b_curr = params["b_" + str(i)]
        U_input_curr = params["U_input_" + str(i)]
        U_prior_activation = params["U_prior_activation_" + str(i)]
        if structures[i]["activation"] == "relu":
            activation_func = relu_backward
        elif structures[i]["activation"] == "sigmoid":
            activation_func = sigmoid_backward
        else:
            print("Not suppported activation func")
            exit
#         set_trace()
        dz_prev, gradient_W, gradient_b = single_layer_backward(dz_prev, U_input_curr, U_prior_activation, W_curr, activation_func)
#         set_trace()
#         params["W_" + str(i)] = W_curr
        grads_table["gradient_W_" + str(i)] = gradient_W
        grads_table["gradient_b_" + str(i)] = gradient_b
        i -= 1
```

### Update the parameters

- Go through each layer to update the parameters using grads_table


```python
def update_weights(depth, params, grads_table, lr):
    for i in range(depth):
        params["W_" + str(i)] = params["W_" + str(i)] - grads_table["gradient_W_" + str(i)] * lr
        params["b_" + str(i)] = params["b_" + str(i)] - grads_table["gradient_b_" + str(i)] * lr
    return params
```

### In practice


```python
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
X, y = make_moons(n_samples = 1000, noise=0.2, random_state=100)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
 
params = init_layers(structures)
test = params.copy()
grads_table = {}
deepth = len(structures)
for epoch in range(10000):
    output = feedforward(X_train, structures, params)
#     print("logloss: {}; accuracy: {}".format(negativelogloss(output, y_train), get_accuracy(y_train, output)))
    backward(output, y_train, structures, params)
    params = update_weights(deepth, params, grads_table, 0.01)
 
y_hat = feedforward(X_test, structures, params)
print("logloss: {}; accuracy: {}".format(negativelogloss(y_hat, y_test), get_accuracy(y_test, y_hat)))
```

 

Thanks to [this post](https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)