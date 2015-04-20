import numpy as np
from mlp import MLP

def test_linear_layers_update():
    n_neurons = [2,3,1]
    x = np.array([[1,2]])
    t = np.array([[2]])
    lr = 0.1
    weights_t = [np.array([[-0.1, -0.1, -0.1],
                  [-1.2, -1.2, -1.2]]),
                 np.array([[-3.4], [-3.4], [-3.4]])
                ]

    print "Test with linear activation function"
    activation_function = 'linear'
    initialization='ones'
    model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)
    print model
    print "Forward pass with input {0}".format(x)
    model.forward(x)
    print "Activations \n{0}".format(model.activations)
    print "Zetas \n{0}".format(model.zetas)
    print "Compute deltas for output {0}".format(t)
    model.compute_deltas(t)
    print "Deltas \n{0}".format(model.deltas)
    print "Update"
    model.update(x,lr)
    print model
    for i in range(len(weights_t)):
        np.testing.assert_almost_equal(weights_t[i], model.weights[i])

def test_tanh_layers_update():
    n_neurons = [2,3,1]
    x = np.array([[1,2]])
    t = np.array([[2]])
    lr = 0.1
    zetas = [np.array([[np.tanh(3+1), np.tanh(3+1), np.tanh(3+1)]]),
             np.array([[np.tanh(np.tanh(4)*3+1)]])]
    deltas = [np.array([[-0.00134185, -0.00134185, -0.00134185]]),
              np.array([[-1.00067342]])]
    weights_t = [np.array([[1.0001342, 1.0001342, 1.0001342],
                          [1.00026837, 1.00026837, 1.00026837]]),
                 np.array([[1.10000023],
                           [1.10000023],
                           [1.10000023]])
                ]

    print "Test with linear activation function"
    activation_function = 'tanh'
    initialization='ones'
    model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)
    print model
    print "Forward pass with input {0}".format(x)
    model.forward(x)
    for i in range(len(zetas)):
        np.testing.assert_almost_equal(zetas[i], model.zetas[i])
    print "Compute deltas for output {0}".format(t)
    model.compute_deltas(t)
    print "Deltas \n{0}".format(model.deltas)
    for i in range(len(deltas)):
        np.testing.assert_almost_equal(deltas[i], model.deltas[i])
    print "Update"
    model.update(x,lr)
    print model
    for i in range(len(weights_t)):
        np.testing.assert_almost_equal(weights_t[i], model.weights[i])

def test_multiple_outputs():
    n_neurons = [2,3,2]
    x = np.array([[1,2]])
    t = np.array([[2,2]])
    lr = 0.1
    zetas = [np.array([[3+1, 3+1, 3+1]]),
             np.array([[4*3+1, 4*3+1]])]
    deltas = [np.array([[2*(4*3+1 -2), 2*(4*3+1 -2), 2*(4*3+1 -2)]]),
              np.array([[4*3+1 -2, 4*3+1 -2]])]
    weights_t = [np.array([[-1.2, -1.2, -1.2],
                  [-3.4, -3.4, -3.4]]),
                 np.array([[-3.4, -3.4],
                           [-3.4, -3.4],
                           [-3.4, -3.4]])
                ]

    print "Test with linear activation function"
    activation_function = 'linear'
    initialization='ones'
    model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)
    print model
    print "Forward pass with input {0}".format(x)
    model.forward(x)
    for i in range(len(zetas)):
        np.testing.assert_almost_equal(zetas[i], model.zetas[i])
    print "Activations \n{0}".format(model.activations)
    print "Zetas \n{0}".format(model.zetas)
    print "Compute deltas for output {0}".format(t)
    model.compute_deltas(t)
    for i in range(len(deltas)):
        np.testing.assert_almost_equal(deltas[i], model.deltas[i])
    print "Deltas \n{0}".format(model.deltas)
    print "Update"
    model.update(x,lr)
    print model
    for i in range(len(weights_t)):
        np.testing.assert_almost_equal(weights_t[i], model.weights[i])


def test_multiple_samples():
    n_neurons = [2,3,1]
    x = np.array([[1,2], [1,2]])
    t = np.array([[2], [2]])
    lr = 0.1
    weights_t = [np.array([[-0.1, -0.1, -0.1],
                  [-1.2, -1.2, -1.2]]),
                 np.array([[-3.4], [-3.4], [-3.4]])
                ]

    print "Test with linear activation function"
    activation_function = 'linear'
    initialization='ones'
    model = MLP(n_neurons, activation_function=activation_function,
                initialization=initialization)
    print model
    print "Forward pass with input {0}".format(x)
    model.forward(x)
    print "Activations \n{0}".format(model.activations)
    print "Zetas \n{0}".format(model.zetas)
    print "Compute deltas for output {0}".format(t)
    model.compute_deltas(t)
    print "Deltas \n{0}".format(model.deltas)
    print "Update"
    model.update(x,lr)
    print model
    for i in range(len(weights_t)):
        np.testing.assert_almost_equal(weights_t[i], model.weights[i])


