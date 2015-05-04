#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
from mlp import MLP

FIGURE_TRAINING=99

def update_plot(x,t,grid,y,plt):
    """
    Updates the image of the actual plot with the original target
    and the prediction of the model
    """
    plt.clf()
    plt.scatter(x,t,color='red')
    plt.plot(grid,y)
    display.clear_output(wait=True)
    display.display(pltgcf())


def generate_regression_data(n_samples=50, bias=1, noise=0, pattern='linear'):
    x = np.random.uniform(-1,1,(n_samples,1))*np.pi

    if pattern == 'sin':
        y = np.sin(x)
    elif pattern == 'linear':
        y = np.copy(x)
    elif pattern == 'sin_cos':
        y = np.sin(np.cos(x)*np.pi)
    else:
        y = np.copy(x)

    if noise > 0:
        y += np.random.normal(size=(n_samples,1), loc=bias, scale=noise)
    else:
        y += bias

    return x,y

def generate_classification_data(n_samples=50, bias=1, noise=0, pattern='binary'):
    mu_true = [2, 8]
    sigma_true = [1, 2]
    psi_true = .4
    y = np.random.binomial(1, psi_true, n_samples)
    x = [np.random.normal(mu_true[i], sigma_true[i]) for i in y]
    y = np.reshape(y,(n_samples,1))
    x = np.reshape(x,(n_samples,1))

    return x,y


def train(model,x,t,lr=0.01,epochs=1,algorithm='sgd',minibatch=1):
    error = np.zeros(epochs)
    for i in range(epochs):
        if algorithm == 'sgd':
            for j in range(len(x)):
                model.forward(x[j])
                model.compute_deltas(t[j])
                model.update(x[j],lr)
                error[i] += model.error(x[j],t[j])
        elif algorithm == 'batch':
            model.forward(x)
            model.compute_deltas(t)
            model.update(x,lr)
            error[i] = model.error(x,t)
        elif algorithm == 'minibatch':
            for j in range((len(x)/minibatch)):
                position = j*minibatch
                model.forward(x[position:position+minibatch])
                model.compute_deltas(t[position:position+minibatch])
                model.update(x[position:position+minibatch],lr)
                error[i] += model.error(x[position:position+minibatch],
                                        t[position:position+minibatch])
    return error

def plot_data(x,y):
    plt.figure()
    plt.scatter(x, y, color='red', label='samples')
    plt.ylabel('target')
    plt.xlabel('x')
    plt.legend()
    plt.show(block=False)

def plot_error(error):
    plt.figure()
    plt.plot(error)
    plt.show()

def plot_performance(model, x, t, update=True):
    grid = np.reshape(np.linspace(np.min(x), np.max(x), 100),(100,1))
    y = model.output(grid)
    plt.figure(FIGURE_TRAINING)
    plt.scatter(x,t)
    plt.plot(grid,y)
    plt.show(block=False)

def test_01():
    # DATA config
    train_algorithm = 'minibatch' #sgd, batch and minibatch
    minibatch = 100 #sgd, batch and minibatch
    pattern = 'sin_cos'
    n_samples = 100
    noise = 0
    bias = 0
    # MODEL config
    n_neurons = [1,6,1]
    activation_function = 'tanh'
    initialization='rand'
    # TRAINING config
    lr = 0.1
    epochs = 100

    x,t = generate_regression_data(n_samples=n_samples, noise=noise,
                        bias=bias,pattern=pattern)
    #plot_data(x,t)

    model = MLP(n_neurons, activation_function=activation_function,
            initialization=initialization)

    error = np.zeros(epochs)
    plt.ion()
    print "Training {0} epochs".format(epochs)
    grid = np.reshape(np.linspace(np.min(x), np.max(x), 100),(100,1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_std = np.std(t)
    ax.set_ylim(np.min(t)-t_std,np.max(t)+t_std)
    ax.scatter(x,t)
    y = model.output(grid)
    line1, = ax.plot(grid,y)
    for i in range(epochs):
        error[i] = train(model, x=x, t=t, lr=lr, epochs=1,
                         algorithm=train_algorithm,
                         minibatch=minibatch)
        y = model.output(grid)
        line1.set_ydata(y)
        fig.canvas.draw()
        time.sleep(0.001)


    plt.figure()
    plt.plot(error)
    plt.show(block=True)

def test_02():
    # DATA config
    train_algorithm = 'minibatch' #sgd, batch and minibatch
    minibatches = [1,2,5,10,20,50,100] #sgd, batch and minibatch
    pattern = 'sin_cos'
    n_samples = 100
    noise = 0
    bias = 0
    # MODEL config
    n_neurons = [1,10,1]
    activation_function = 'tanh'
    initialization='rand'
    # TRAINING config
    lr = 0.1
    epochs = 500

    x,t = generate_regression_data(n_samples=n_samples, noise=noise,
                        bias=bias,pattern=pattern)

    error = np.zeros((len(minibatches),epochs))
    for i, minibatch in enumerate(minibatches):
        model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)

        print "Training {0} epochs".format(epochs)
        grid = np.reshape(np.linspace(np.min(x), np.max(x), 100),(100,1))
        y = model.output(grid)
        for j in range(epochs):
            error[i,j] = train(model, x=x, t=t, lr=lr, epochs=1,
                            algorithm=train_algorithm,
                            minibatch=minibatch)
            y = model.output(grid)

    plt.figure()
    plt.plot(error.T)
    plt.legend(minibatches)
    plt.show(block=True)

def test_03():
    # DATA config
    train_algorithm = 'minibatch' #sgd, batch and minibatch
    minibatch = 20 #sgd, batch and minibatch
    pattern = 'binary'
    n_samples = 100
    noise = 0
    bias = 0
    # MODEL config
    n_neurons = [1,2,1]
    activation_function = 'tanh'
    activation_out_function = 'linear'
    initialization='rand'
    # TRAINING config
    lr = 0.1
    epochs = 500
    error_function = 'ce'

    x,t = generate_classification_data(n_samples=n_samples, noise=noise,
                                       pattern=pattern)

    model = MLP(n_neurons, activation_function=activation_function,
                activation_out_function=activation_out_function,
                initialization=initialization, error_function=error_function)

    error = np.zeros(epochs)
    plt.ion()
    print "Training {0} epochs".format(epochs)
    grid = np.reshape(np.linspace(np.min(x), np.max(x), 100),(100,1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t_std = np.std(t)
    ax.set_ylim(np.min(t)-t_std,np.max(t)+t_std)
    ax.scatter(x,t)
    y = model.output(grid)
    line1, = ax.plot(grid,y)
    for i in range(epochs):
        error[i] = train(model, x=x, t=t, lr=lr, epochs=1,
                         algorithm=train_algorithm,
                         minibatch=minibatch)
        y = model.output(grid)
        line1.set_ydata(y)
        fig.canvas.draw()
        time.sleep(0.001)

    plt.figure()
    plt.plot(error)
    plt.show(block=True)


def main():
    test_03()

if __name__ == "__main__":
    main()
