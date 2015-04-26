#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


class MLP:
    """
    Implementation based on the book:
    [Bishop2006] Bishop, Christopher M. Pattern recognition and machine learning. Vol. 1. New York: springer, 2006.
    """
    def __init__(self, n_neurons, activation_function='tanh',
                 activation_out_function='linear',
                 initialization='ones'):
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)-1
        self.weights = []
        self.biases = []
        self.activations = []
        self.zetas = []
        self.deltas = []
        self.activation_function = activation_function

        for i in range(self.n_layers):
            self.weights.append([])
            self.biases.append([])
            self.activations.append([])
            self.zetas.append([])
            self.deltas.append([])

        if activation_function == 'tanh':
            self.activation_f = lambda x: np.tanh(x)
            self.activation_fdx = lambda x: (1 - np.tanh(x)**2)
        elif activation_function == 'linear':
            self.activation_f = lambda x: x
            self.activation_fdx = lambda x: np.ones(x.shape)

        if activation_out_function == 'tanh':
            self.activation_out_f = lambda x: np.tanh(x)
        elif activation_out_function == 'linear':
            self.activation_out_f = lambda x: x

        self.initialize(initialization)

    def initialize(self, initialization='ones'):
        if initialization == 'zeros':
            for i in range(self.n_layers):
                self.weights[i] = np.zeros((self.n_neurons[i],
                                            self.n_neurons[i+1]))
                self.biases[i] = np.zeros((1, self.n_neurons[i+1]))
        elif initialization == 'ones':
            for i in range(self.n_layers):
                self.weights[i] = np.ones((self.n_neurons[i],
                                           self.n_neurons[i+1]))
                self.biases[i] = np.ones((1, self.n_neurons[i+1]))
        elif initialization == 'rand':
            for i in range(self.n_layers):
                if self.activation_function == 'tanh':
                    hi_limit = 6/np.sqrt(self.n_neurons[i] + self.n_neurons[i+1])
                else:
                    hi_limit = 0.5
                self.weights[i] = np.random.uniform(low=-hi_limit, high=hi_limit,
                        size=(self.n_neurons[i], self.n_neurons[i+1]))
                self.biases[i] = np.random.uniform(low=-hi_limit,
                        high=hi_limit, size=(1, self.n_neurons[i+1]))

    def forward(self, x):
        """
        Forward pass
        eq. (5.62), (5.63), (5.64)
        """
        self.activations[0] = np.dot(x,self.weights[0]) + self.biases[0]
        self.zetas[0] = self.activation_f(self.activations[0])
        for i in range(1, self.n_layers-1):
            self.activations[i] = np.dot(self.zetas[i-1],self.weights[i]) + self.biases[i]
            self.zetas[i] = self.activation_f(self.activations[i])
        self.activations[-1] = np.dot(self.zetas[-2],self.weights[-1]) + self.biases[-1]
        self.zetas[-1] = self.activation_out_f(self.activations[-1])
        return self.zetas[-1]

    def compute_deltas(self, target):
        """
        Eq. (5.56)
        It do not compute the delta for the hidden bias.
        As it is not used on the backpropagation
        """
        self.deltas[-1] = self.zetas[-1] - target
        for i in reversed(range(self.n_layers-1)):
            self.deltas[i] = np.multiply(np.dot(self.deltas[i+1], self.weights[i+1].T),
                                         self.activation_fdx(self.activations[i]))
        return self.deltas

    def update(self, x, learning_rate):
        n_samples = x.shape[0]
        self.weights[0] -= learning_rate*np.dot(x.T, self.deltas[0])/n_samples
        self.biases[0] -= np.average(learning_rate*self.deltas[0], axis=0)
        for i in range(1,self.n_layers):
            self.weights[i] -= learning_rate*np.dot(self.zetas[i-1].T,
                                                    self.deltas[i])/n_samples
            self.biases[i] -= np.average(learning_rate*self.deltas[i], axis=0)

    def error(self, x, t):
        self.forward(x)
        y = self.zetas[-1]
        return np.sum(np.subtract(y,t)**2)/2

    def __str__(self):
        return ("ANN architecture\n"
                    "n_neurons = {0}\n"
                    "n_layers = {1}\n"
                    "weights = {2}\n"
                    "biases = {3}\n"
                    "activation_function = {4}\n").format(
                            self.n_neurons, self.n_layers, self.weights,
                            self.biases, self.activation_f)

    #--------------------------------------
    # Not implemented yet
    #

    def test(self, X):
        y = np.zeros(np.size(X))
        for i, x in enumerate(X):
            y[i] = self.output(x)
        return y

    def feature_extraction(self, X):
        z_hidden = np.zeros((np.size(X), self.n_hidden + 1))
        for i, x in enumerate(X):
            z_hidden[i] = self.hidden_output(x)
        return z_hidden

    def mean_error(self, X, t):
        return self.error(X,t)/np.size(t)

    def hidden_output(self, x):
        [a_hidden, z_hidden, y] = self.forward(x)
        return z_hidden

    def output(self, x):
        self.forward(x)
        return self.zetas[-1]

    def all_output(self, x):
        return self.forward(x)

def main():
    n_neurons = [2,3,2]
    x = np.array([[1,2], [1,2], [1,2]])
    t = np.array([[2,4], [2,4], [2,4]])
    lr = 0.1
    epochs = 200

    activation_function = 'linear'
    initialization='ones'
    model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)
    print model
    error = np.zeros(epochs)
    for i in range(epochs):
        model.forward(x)
        print "Forward pass with input {0} gives output {1}".format(x, model.zetas[-1])
        model.compute_deltas(t)
        model.update(x,lr)
        print model.weights
        error[i] = model.error(x,t)
        print error[i]
    plt.plot(error)
    plt.show()
    print model

if __name__ == "__main__":
    main()
