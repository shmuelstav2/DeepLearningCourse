import sys
sys.path.append('/data/anaconda/envs/py35/lib/python3.5/site-packages')


import pandas as pd
from math import ceil
import numpy as np
import os
import tempfile

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import time


class EX1_FFN(BaseEstimator, ClassifierMixin):

    def __init__(self, layers_dims=None, learning_rate=0.09, num_iterations=10000,
                 batch_size=1000, parameters=None, use_batchnorm=False, cost_th=1e-4,dropout=False):

        self.costs_train = []
        self.costs_val = []
        self.accuracies_train = []
        self.accuracies_val = []

        self.layers_dims = layers_dims
        self.parameters = self.initialize_parameters(layers_dims) if parameters is None else parameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.use_batchnorm = use_batchnorm
        self.cost_th = cost_th
        self.dropout= dropout
        #fig, ax = plt.subplots()
    def fit(self, X, y=None):
        parameters, costs_train, costs_val, \
        accuracies_train, accuracies_val = self.L_layer_model(X, y, self.layers_dims, self.learning_rate,
                                                              self.num_iterations, self.batch_size, self.use_batchnorm,
                                                              self.parameters, self.cost_th,self.dropout)

        self.parameters = parameters
        self.costs_train += costs_train
        self.costs_val += costs_val
        self.accuracies_val += accuracies_val
        self.accuracies_train += accuracies_train

        return self

    def predict(self, X, y=None,dropout = False):
        AL, caches = self.L_model_forward(X, self.parameters, self.use_batchnorm,dropout = dropout)
        y_pred = np.argmax(AL, axis=0)
        assert X.shape[1] == y_pred.shape[0], "wrong size of y_pred. X shape: {} y_pred shape: {}".format(X.shape,
                                                                                                          y_pred.shape)
        return y_pred

    def score(self, X, y=None):
        AL, caches = self.L_model_forward(X, self.parameters, self.use_batchnorm,dropout = dropout)
        return self.compute_cost(AL, y)

    def initialize_parameters(self, layer_dims, w_scale=1):
        """
        input: an array of the dimensions of each layer in the network (layer 0 is the size of the
        flattened input, layer L is the output sigmoid)
        output: a dictionary containing the initialized W and b parameters of each layer
        (W1…WL, b1…bL).
        Hint: Use the randn and zeros functions of numpy to initialize W and b, respectively
        """
        W = []
        b = []
        for i in range(len(layer_dims) - 1):
            W.append(np.random.randn(layer_dims[i], layer_dims[i + 1]) * w_scale)
            b.append(np.zeros(layer_dims[i + 1]))
        return {'W': W,
                'b': b}

    def linear_forward(self, A, W, b):
        """
        linear_forward(A, W, b)
        Description: Implement the linear part of a layer's forward propagation.
        input:
        A – the activations of the previous layer
        W – the weight matrix of the current layer (of shape [size of current layer, size of
        previous layer])
        B – the bias vector of the current layer (of shape [size of current layer, 1])
        Output:
        Z – the linear component of the activation function (i.e., the value before applying the
        non-linear function)
        linear_cache – a dictionary containing A, W, b (stored for making the backpropagation
        easier to compute)
        """

        Z = np.dot(W.T, A)
        samples_dim = Z.shape[1]
        B = np.tile(b, samples_dim).reshape(samples_dim, b.shape[0]).T
        Z += B

        linear_cache = {"A": A,
                        "W": W,
                        "b": b}

        return Z, linear_cache

    def sigmoid(self, Z):
        """
        Z – the linear component of the activation function
        Output:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation
        """
        A = 1.0 / (1.0 + np.exp(-Z))
        activation_cache = Z
        return A, activation_cache

    def relu(self, Z):
        """
        Input:
        Z – the linear component of the activation function
        Output:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagatio
        """
        A = np.maximum(Z, 0)
        activation_cache = Z
        return A, activation_cache

    def stable_softmax(self, Z):
        """
        softmax(Z)
        Input:
        Z – the linear component of the activation function
        Output:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation

        ::
        """

        activation_cache = Z
        np.subtract(Z , np.amax(Z,axis=0))
        exps = np.exp(np.subtract(Z , np.amax(Z,axis=0)))
        #exps = np.exp(Z - np.max(Z))
        #exps = np.exp(Z)
        sum_vec = np.sum(exps, axis=0)
        if np.min(sum_vec) == 0:
            print("sum vec error")
            print(sum_vec)
        return (exps / (sum_vec)), activation_cache

    def linear_activation_forward(self, A_prev, W, B, activation ,dropout):
        """
        Description:
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        Input:
        A_prev – activations of the previous layer
        W – the weights matrix of the current layer
        B – the bias vector of the current layer
        Activation – the activation function to be used (a string, either “sigmoid” or “relu”)
        Output:
        A – the activations of the current layer
        cache – a joint dictionary containing both linear_cache and activation_cache
        """



        activation_l = activation.lower()
        assert activation_l == "softmax" or activation_l == "relu", "wrong activation function"
        Z, linear_cache = self.linear_forward(A_prev, W, B)
        A, activation_cache = self.stable_softmax(Z) if activation_l == "softmax" else self.relu(Z)
        D = None
        if dropout:
         D = np.random.rand(A.shape[0], A.shape[1])
         D = D < 0.9
         A = np.multiply(A, D)
         A = A / 0.9
        cache = {"linear_cache": linear_cache,
                 "activation_cache": activation_cache,
                 "dropout":D}
        return A, cache

    def apply_batchnorm(self, X):
        """
        Description:
        performs batchnorm on the received activation values of a given layer.
        Input:
        A - the activation values of a given layer
        output:
        NA - the normalized activation values, based on the formula learned in class
        """
        A = X.T
        mu = np.mean(A, axis=0)
        var = np.var(A, axis=0)

        NA = (A - mu) / np.sqrt(var + 1e-8)

        return NA.T

    def L_model_forward(self, X, parameters, use_batchnorm ,dropout):
        """
        Description:
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
        computation
        Input:
        X – the data, numpy array of shape (input size, number of examples)
        parameters – the initialized W and b parameters of each layer
        use_batchnorm - a boolean flag used to determine whether to apply batchnorm after
        the activation (note that this option needs to be set to “false” in Section 3 and “true” in
        Section 4).
        Output:
        AL – the last post-activation value
        caches – a list of all the cache objects generated by the linear_forward function
        """
        W = parameters["W"]
        B = parameters["b"]
        A = X
        caches = []
        L = len(B)
        for i in range(L - 1):
            if use_batchnorm:
                A = self.apply_batchnorm(A)
            dropout_layer = False
            if (i==1 and dropout==True):
                dropout_layer = True
            A, cache = self.linear_activation_forward(A, W[i], B[i], "relu",dropout_layer)
            caches.append(cache)


        AL, cache = self.linear_activation_forward(A, W[L - 1], B[L - 1], "softmax",dropout_layer)
        caches.append(cache)
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Description:
        Implement the cost function defined by equation. (see the slides of the first lecture for
        additional information if needed).
        𝑐𝑜𝑠𝑡 = − '
        ( ∗ Σ [,𝑦. ∗ log(𝐴𝐿)6 + (,1 − 𝑦.6 ∗ log(1 − 𝐴𝐿))] ('
        Input:
        AL – probability vector corresponding to your label predictions, shape (1, number of
        examples)
        Y – the labels vector (i.e. the ground truth)
        Output:
        cost – the cross-entropy cost
        """

        # samples_dim = Z.shape[1]
        # B = np.tile(b, samples_dim).reshape(samples_dim, b.shape[0]).T
        # Z += B
        #
        # eps=1e-16
        #
        # y_pred = np.maximum(AL, eps)
        # y_pred = np.minimum(y_pred, (1 - eps))
        # return -(np.sum(Y * np.log(y_pred)) + np.sum((1 - Y) * np.log(1 - y_pred))) / len(Y)

        m = Y.shape[0]
        p = AL.T
        log_likelihood = -np.log(p[range(m), Y])
        loss = np.sum(log_likelihood) / m
        print("loss: {}".format(loss))
        return loss

    def Linear_backward(self, dZ, cache,dropout,D ):
        """
        Implements the linear part of the backward propagation process for a single layer
        Input:
        dZ – the gradient of the cost with respect to the linear output of the current layer (layer
        l)
        cache – tuple of values (A_prev, W, b) coming from the forward propagation in the
        current layer
        Output:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1),
        same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dA_prev = np.dot(W, dZ)
        if dropout:
            dA_prev = np.multiply(dA_prev,D)
            dA_prev = dA_prev/0.9
        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)

        return dA_prev, dW, db

    def reluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def sigmoidDerivative(self, x):
        return self.sigmoid(x)[0] * (1 - self.sigmoid(x)[0])

    def linear_activation_backward(self, dA, cache,dropout,activation):
        """
        Description:
        Implements the backward propagation for the LINEAR->ACTIVATION layer. The function
        first computes dZ and then applies the linear_backward function.
        Some comments:
        • The derivative of ReLU is 𝑓5(𝑥) = 71 𝑖𝑓 𝑥 > 0
        0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
        • The Sigmoid function is 𝜎(𝑥) = '
        'BCDE and its derivative is
        𝜎5(𝑥) = 𝜎(𝑥) ∗ (1 − 𝜎(𝑥))
        • You should use the activations cache created earlier for the calculation of the
        activation derivative and the linear cache should be fed to the linear_backward
        function
        Input:
        dA – post activation gradient of the current layer
        cache – contains both the linear cache and the activations cache
        Output:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1),
        same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b
        """
        activation_l = activation.lower()
        assert activation_l == "softmax" or activation_l == "relu", "wrong activation function"
        linear_cache = cache["linear_cache"]
        activation_cache = cache["activation_cache"]
        D =None
        if dropout:
            D = cache["dropout"]
        A_prev = linear_cache["A"]
        W = linear_cache["W"]
        b = linear_cache["b"]

        dZ = self.relu_backward(dA, activation_cache) if activation_l == "relu" else self.softmax_backward(dA,
                                                                                                           activation_cache)
        cache = A_prev, W, b
        dA_prev, dW, db = self.Linear_backward(dZ, cache,dropout,D)
        return dA_prev, dW, db

    def relu_backward(self, dA, activation_cache):
        """
        Description:
        Implements backward propagation for a ReLU unit
        Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
        Output:
        dZ – gradient of the cost with respect to Z
        """
        Z = activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax_backward(self, dA, activation_cache):
        """
        Description:
        Implements backward propagation for a softmax unit
        Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
        Output:
        dZ – gradient of the cost with respect to Z
        """
        Z = activation_cache
        return dA

    def sigmoid_backward(self, dA, activation_cache):
        """
        Description:
        Implements backward propagation for a sigmoid unit
        Input:
        dA – the post-activation gradient
        activation_cache – contains Z (stored during the forward propagation)
        Output:
        dZ – gradient of the cost with respect to Z
        """
        Z = activation_cache
        dZ = dA * self.sigmoidDerivative(Z)
        return dZ

    def L_model_backward(self, AL, Y, caches,dropout):
        """
        Description:
        Implement the backward propagation process for the entire network.
        Some comments:
        the backpropagation for the sigmoid function should be done only once as only the
        output layers uses it and the RELU should be done iteratively over all the remaining
        layers of the network.
        The derivative for the output of the softmax layer can be calculated using:
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        Input:
        AL - the probabilities vector, the output of the forward propagation (L_model_forward)
        Y - the true labels vector (the "ground truth" - true classifications)
        Caches - list of caches containing for each layer: a) the linear cache; b) the activation
        cache
        Output:
        Grads - a dictionary with the gradients
        grads["dA" + str(l)] = ...
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ...
        """
        L = len(caches)  # the number of layers
        grads = {}

        #########################
        # Softmax layer
        #########################
        # m = Y.shape[0]
        # dAL = stable_softmax(AL.T)
        # dAL[range(m), Y] -= 1
        # dAL = dAL.T / m

        out_size = len(caches[-1]['linear_cache']["b"])
        Y_vec = np.zeros((out_size, len(Y)))
        for i in range(len(Y)):
            Y_vec[Y[i], i] = 1

        # Initializing the backpropagation
        dAL = AL - Y_vec
        dropout_layer = False
        current_cache = caches[L - 1]

        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                           current_cache,dropout_layer,
                                                                                                           activation="softmax")

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            dropout_layer = False
            current_cache = caches[l]
            if (l==2 and dropout ==True):
                dropout_layer = True
                Dcahe = caches[l - 1]
                current_cache["dropout"] = Dcahe["dropout"]

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache,dropout_layer,
                                                                             activation="relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def Update_parameters(self, parameters, grads, learning_rate):
        """
        Description:
        Updates parameters using gradient descent
        Input:
        parameters – a python dictionary containing the DNN architecture’s parameters
        grads – a python dictionary containing the gradients (generated by L_model_backward)
        learning_rate – the learning rate used to update the parameters (the “alpha”)
        Output:
        parameters – the updated values of the parameters object provided as input
        """
        L = len(parameters["b"])

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W"][l] = parameters["W"][l] - learning_rate * grads["dW" + str(l + 1)].T
            parameters["b"][l] = parameters["b"][l] - learning_rate * grads["db" + str(l + 1)].reshape(
                parameters["b"][l].shape)

        return parameters

    def L_layer_model(self, X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False,
                      parameters=None, cost_th=1e-4,dropout=False):
        """
        Description:
        Implements a L-layer neural network. All layers but the last should have the ReLU
        activation function, and the final layer will apply the sigmoid activation function. The
        size of the output layer should be equal to the number of labels in the data. Please
        select a batch size that enables your code to run well (i.e. no memory overflows while
        still running relatively fast).
        Hint: the function should use the earlier functions in the following order: initialize ->
        L_model_forward -> compute_cost -> L_model_backward -> update parameters
        Input:
        X – the input data, a numpy array of shape (height*width , number_of_examples)
        Comment: since the input is in grayscale we only have height and width, otherwise it
        would have been height*width*3
        Y – the “real” labels of the data, a vector of shape (num_of_classes, number of
        examples)
        Layer_dims – a list containing the dimensions of each layer, including the input
        batch_size – the number of examples in a single training batch.
        Output:
        parameters – the parameters learnt by the system during the training (the same
        parameters that were updated in the update_parameters function).
        costs – the values of the cost function (calculated by the compute_cost function). One
        value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).
        """
        costs_train = [10]
        costs_val = [10]
        accuracies_train = []
        accuracies_val = []

        x_train, x_val, y_train, y_val = train_test_split(X.T, Y, test_size=0.2, random_state=42)
        x_train = x_train.T
        x_val = x_val.T

        examples = x_train.shape[1]
        batch_itr = ceil(examples / batch_size)
        for k in range(num_iterations):

            for l in range(batch_itr):
                X_batch = x_train[:, l * batch_size:(l + 1) * batch_size]
                Y_batch = y_train[l * batch_size:(l + 1) * batch_size]

                AL, caches = self.L_model_forward(X_batch, parameters, use_batchnorm=use_batchnorm,dropout = dropout)

                grads = self.L_model_backward(AL, Y_batch, caches,dropout = dropout)
                parameters = self.Update_parameters(parameters, grads, learning_rate)
            if k % 100 == 0:
                print("iteration: {}".format(k))
                AL, caches = self.L_model_forward(x_train, parameters, use_batchnorm=use_batchnorm,dropout = dropout)
                cost = self.compute_cost(AL, y_train)
                costs_train.append(cost)
                print(AL[:, 1])
                print("train cost: {}".format(cost))

                AL, caches = self.L_model_forward(x_val, parameters, use_batchnorm=use_batchnorm,dropout = dropout)
                cost = self.compute_cost(AL, y_val)
                costs_val.append(cost)
                print("val cost: {}".format(cost))

                y_pred_train = self.predict(x_train, y_train)
                y_pred_val = self.predict(x_val, y_val)
                acc_train = accuracy_score(y_train, (np.rint(y_pred_train)).astype(int))
                acc_val = accuracy_score(y_val, (np.rint(y_pred_val)).astype(int))
                accuracies_train.append(acc_train)
                accuracies_val.append(acc_val)

                if abs(costs_val[-2] - cost) < cost_th:
                    print("*********** Reach cost threshold **********")
                    return parameters, costs_train[1:], costs_val[1:], accuracies_train, accuracies_val

        return parameters, costs_train[1:], costs_val[1:], accuracies_train, accuracies_val


def input_flatten(x):
    x = x.astype(np.float32) / 255
    x = np.expand_dims(x, -1)
    return x.reshape((x.shape[0], 784)).T


def plot_loss(cost_train, cost_val, accuracies_train, accuracies_val):
    #fig = plt.figure(figsize=(20, 30))
    fig, ax = plt.subplots(figsize=(20, 30))
    fig.suptitle('Log Loss over iterations')

    ax = fig.add_subplot(4, 1, 1)
    ax.plot(cost_train)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Log Loss train')

    ax = fig.add_subplot(4, 1, 2)
    ax.plot(cost_val)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Log Loss validation')

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(accuracies_train)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Accuracy train')

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(accuracies_val)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Accuracy val')
    plt.savefig('/data/home/shmuelstav/Moving Average dropout.png')


def check_batch_size ():
    """
    check the opt batch size (running time vs performance)
    :return:
    """
    lables = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_f = input_flatten(x_train)
    x_test_f = input_flatten(x_test)

    bs = np.geomspace(100, x_train_f.shape[1]*0.8, num=20)
    bs = bs.astype(int)
    print (bs)
    et = []
    layers_dims = [x_train_f.shape[0], 20, 7, 5, lables]

    for batchsize in bs:
        no_batch_norm = EX1_FFN(layers_dims=layers_dims, learning_rate=0.09, num_iterations=10,
                                batch_size=batchsize, parameters=None, use_batchnorm=False, cost_th=1e-4)
        start = time.time()
        no_batch_norm.fit(x_train_f, y_train)
        end = time.time()
        print ("batch_size: {}     time: {}".format(batchsize, end-start))
        et.append(end-start)

    plt.plot(bs, et)
    plt.grid(True)


def main ():
    #############################################
    # Loading the mnist  data
    #############################################
    lables = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_f = input_flatten (x_train)
    x_test_f = input_flatten (x_test)

    layers_dims = [x_train_f.shape[0], 20, 7, 5, lables]

    #############################################
    # Checking the batch optimum size
    #############################################
    #check_batch_size()

    #############################################
    # no batch norm.
    #############################################
    '''
    no_batch_norm = EX1_FFN (layers_dims=layers_dims,  learning_rate=0.09, num_iterations=2000,
                 batch_size=1000, parameters=None, use_batchnorm=False, cost_th=1e-4,dropout=False)
    no_batch_norm.fit(x_train_f, y_train)

    plot_loss(no_batch_norm.costs_train, no_batch_norm.costs_val, no_batch_norm.accuracies_train, no_batch_norm.accuracies_val)

    result = pd.DataFrame([no_batch_norm.costs_train, no_batch_norm.costs_val, no_batch_norm.accuracies_train,
                           no_batch_norm.accuracies_val]).transpose()
    result.columns = ['costs_train', 'costs_val', 'accuracies_train', 'accuracies_val']
    result

    y_pred_test = no_batch_norm.predict(x_test_f, y_test)
    acc_test = accuracy_score(y_test, (np.rint(y_pred_test)).astype(int))
    acc_test

    no_batch_norm.get_params()

    #############################################
    # With batch norm.
    #############################################

    batch_norm = EX1_FFN(layers_dims=layers_dims, learning_rate=0.09, num_iterations=500,
                            batch_size=1000, parameters=None, use_batchnorm=True, cost_th=1e-4,dropout=False)
    batch_norm.fit(x_train_f, y_train)

    plot_loss(batch_norm.costs_train, batch_norm.costs_val, batch_norm.accuracies_train,
              batch_norm.accuracies_val)

    result = pd.DataFrame([batch_norm.costs_train, batch_norm.costs_val, batch_norm.accuracies_train,
                           batch_norm.accuracies_val]).transpose()
    result.columns = ['costs_train', 'costs_val', 'accuracies_train', 'accuracies_val']
    result

    y_pred_test = batch_norm.predict(x_test_f, y_test)
    acc_test = accuracy_score(y_test, (np.rint(y_pred_test)).astype(int))
    acc_test

    batch_norm.get_params()
    '''
    #############################################
    # Dropout
    #############################################
    no_batch_norm = EX1_FFN(layers_dims=layers_dims, learning_rate=0.09, num_iterations=2000,
                            batch_size=100, parameters=None, use_batchnorm=False, cost_th=1e-4,dropout=True)
    no_batch_norm.fit(x_train_f, y_train)

    plot_loss(no_batch_norm.costs_train, no_batch_norm.costs_val, no_batch_norm.accuracies_train,
              no_batch_norm.accuracies_val)

    result = pd.DataFrame([no_batch_norm.costs_train, no_batch_norm.costs_val, no_batch_norm.accuracies_train,
                           no_batch_norm.accuracies_val]).transpose()
    result.columns = ['costs_train', 'costs_val', 'accuracies_train', 'accuracies_val']
    result

    y_pred_test = no_batch_norm.predict(x_test_f, y_test)
    acc_test = accuracy_score(y_test, (np.rint(y_pred_test)).astype(int))
    acc_test

    no_batch_norm.get_params()



if __name__ == "__main__":
    main()


