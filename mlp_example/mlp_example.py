
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


# Multi Layer Perceptron
# ======================
# 
# Implementaton of a MLP with one hidden layer and one linear output unit

# In[3]:

class MLP:
    """
    Implementation based on the book:
    [Bishop2006] Bishop, Christopher M. Pattern recognition and machine learning. Vol. 1. New York: springer, 2006.
    """
    def __init__(self, n_input, n_hidden, n_output, activation='tanh', 
                 learning_rate=0.001, lr_policy='fixed', stepsize=20, 
                 gamma=0.1, low=0.0001, high=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output        
        self.lr = learning_rate
        self.activation = activation
        
        self.lr_policy = lr_policy
        self.gamma = gamma
        self.low = low
        self.high = high
        self.stepsize = stepsize
        self.step = 0
        
        self.initialize()
        
    def initialize(self):
        """
        Initialization based on the article:
        [Xavier10] Y. Bengio, X. Glorot, Understanding the difficulty of training deep feedforward neuralnetworks, AISTATS 2010
        """
        if self.activation == 'linear':
            hi_limit = 0.5
            oh_limit = 0.5
        elif self.activation == 'tanh':
            hi_limit = 6/(np.sqrt(self.n_input + self.n_hidden))
            oh_limit = 0.5
        elif self.activation == 'sin':
            hi_limit = 12/(np.sqrt(self.n_input + self.n_hidden))
            oh_limit = 1
        self.w_hi = np.random.uniform(-hi_limit,hi_limit,size=(n_hidden, (n_input + 1)))
        self.w_oh = np.random.uniform(-oh_limit,oh_limit,size=(n_output, (n_hidden + 1)))

        
    def train(self,X,T):
        for x, t in zip(X,T):
            [a_hidden, z_hidden, y] = self.forward(x)
            [d_hidden, d_output] = self.compute_deltas(y,t,z_hidden)
            self.update(x,d_hidden,z_hidden,d_output)
            
            self.step += 1
            if lr_policy == 'step':
                if self.step%self.stepsize == 0:
                    self.lr *= self.gamma
            elif lr_policy == 'rand':
                if self.step%self.stepsize == 0:
                    self.lr = np.random.uniform(low=self.low, high=self.high)
      
    def test(self, X):
        y = np.zeros(np.size(X))
        for i, x in enumerate(X):
            y[i] = mlp.output(x)
        return y
    
    def feature_extraction(self, X):
        z_hidden = np.zeros((np.size(X), self.n_hidden + 1))
        for i, x in enumerate(X):
            z_hidden[i] = self.hidden_output(x)
        return z_hidden
    
    def error(self, X, t):
        y = self.test(X)
        return np.sum(np.subtract(y,t)**2)/2
    
    def mean_error(self, X, t):
        return self.error(X,t)/np.size(t)
    
    def hidden_output(self, x):
        [a_hidden, z_hidden, y] = self.forward(x)
        return z_hidden
        
    def output(self, x):
        [a_hidden, z_hidden, y] = self.forward(x)
        return y
    
    def all_output(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward pass
        eq. (5.62), (5.63), (5.64)
        """
        input_pattern = np.append(x,1)
        a_hidden = np.dot(self.w_hi, input_pattern)
        if self.activation == 'linear':
            z_hidden = a_hidden
        elif self.activation == 'tanh':
            z_hidden = np.tanh(a_hidden)
        elif self.activation == 'sin':
            z_hidden = np.sin(a_hidden)
        z_hidden = np.append(z_hidden,1)
        y = np.dot(self.w_oh, z_hidden)
        return [a_hidden, z_hidden, y]

    def compute_deltas(self, y, t, z_hidden):
        d_output = self.compute_delta_output(y,t)
        d_hidden = self.compute_delta_hidden(z_hidden,d_output)
        return [d_hidden, d_output]
        
    def compute_delta_hidden(self, z_hidden, d_output):
        """
        Eq. (5.56)
        """
        d_hidden = np.zeros(self.n_hidden)
        for j in range(self.n_hidden):
            # sumation part
            sumation = 0
            for k in range(np.size(d_output)):
                sumation += self.w_oh[k,j]*d_output[k]
            # derivative of activation function
            if self.activation == 'tanh':
                derivative = (1 - z_hidden[j]**2)
            elif self.activation == 'linear':
                derivative = z_hidden[j]
            elif self.activation == 'sin':
                derivative = np.cos(z_hidden[j])
            else:
                print("Unknown activation function")
                exit(0)
            # Multiplication
            d_hidden[j] = derivative*sumation
                
        return d_hidden

    def compute_delta_output(self,y,t):
        return y - t
    
    def update(self, x, d_hidden, z_hidden, d_output):
        x = np.append(x,1)
        #print("n_input = {0}\n"
        #      "n_hidden = {1}\n"
        #      "n_output = {2}\n"
        #      "w_hi size = {3}\n"
        #      "w_oh size = {4}".format(self.n_input, self.n_hidden, self.n_output, 
        #                               self.w_hi.shape, self.w_oh.shape))
        for k in range(self.n_output):
            for j in range(self.n_hidden + 1):
                self.w_oh[k,j] -= self.lr*d_output[k]*z_hidden[j]
                
        for j in range(self.n_hidden):
            for i in range(self.n_input + 1):
                self.w_hi[j,i] -= self.lr*d_hidden[j]*x[i]
    
    def print_architecture(self):
        print(self.string_architecture())
        
    def string_architecture(self):
        string = ("ANN with next architecture\n"
                    "n_input = {0}\n"
                    "n_hidden = {1}\n"
                    "n_output = {2}\n"
                    "lr_policy = {3}\n"
                    "lr = {4}").format(self.n_input,self.n_hidden, self.n_output, self.lr_policy, self.lr)
        if self.lr_policy == 'step':
            string += ("stepsize = {0}\n"
                       "gamma = {1}\n").format(self.stepsize, self.gamma)
        string += "Input to hidden weights\n"
        string += np.array_str(self.w_hi)
        string += "Hidden to output weights\n"
        string += np.array_str(self.w_oh)
        return string


# Auxiliar functions
# ==================
# 
# To visualize and store the results

# In[4]:

def update_plot(X,Y,grid,prediction,plt):
    """
    Updates the image of the actual plot with the original target
    and the prediction of the model
    """
    plt.clf()
    plt.scatter(X,Y,color='red')
    plt.plot(grid,prediction)
    display.clear_output(wait=True)
    display.display(plt.gcf())

def export_plot(plt,filename,format='pdf'):
    """
    Exports the figure into the project folder
    suported formats : eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    """
    formated_filename = "{0}_{1}.{2}".format(time.strftime("%Y%m%d_%H%M%S"), filename, format)
    plt.savefig(formated_filename, format=format)


# Pattern to approximate
# ======================

# In[5]:

N = 100 # datapoints
#X = np.random.rand(N)*np.pi*2
X = np.random.uniform(-1,1,N)*np.pi
#X = np.linspace(-1,1,N)*np.pi
grid = np.linspace(-1,1,150)*(np.pi + 0.5)
#Y = X # linear
Y = np.sin(X)+1 # Sin signal
#Y = np.sin(np.cos(X)*np.pi)+1
#Y = np.abs(X) # Absolute value
#Y = X**2 # Squared
#Y = X > 0

plt.scatter(X,Y,color='red')
plt.ylabel('target')
plt.xlabel('x')
export_plot(plt,filename="target_pattern")


# Especification of the MLP
# =========================
# 
# it shows the initial weights

# In[6]:

n_input = 1
n_hidden = 4
n_output = 1
activation = 'tanh' # 'tanh', 'linear'

# Learning rate parameters
lr_policy = 'step' # 'step', 'rand', 'fixed'
lr = 0.1
gamma = 0.1
low = 0.0001
high = 0.1
stepsize = 40*N

EPOCHS = 120

mlp = MLP(n_input, n_hidden, n_output, activation=activation, learning_rate=lr, 
          lr_policy=lr_policy, stepsize=stepsize, gamma=gamma, low=low, high=high)
mlp.print_architecture()


# Initial prediction
# ==================
# 
# It contains the randomly initialized weights and bias

# In[7]:

plt.clf()
prediction = mlp.test(grid)
plt.scatter(X,Y,color='red')
plt.plot(grid,prediction)

# plot formatting
plt.legend(['prediction','target'])
plt.ylabel('y')
plt.xlabel('x')
export_plot(plt,"initial_prediction")


# Training
# ========

# In[8]:

plt.clf()
error = np.zeros(EPOCHS)
for epoch in range(EPOCHS):
    prediction = mlp.test(grid)
    update_plot(X,Y,grid,prediction,plt)
    error[epoch] = mlp.mean_error(X,Y)
    
    mlp.train(X,Y)
    
plt.legend(['prediction','target'])
plt.ylabel('target')
plt.xlabel('x')
export_plot(plt,"final_prediction")


# Training error
# ==============

# In[9]:

plt.clf()
plt.plot(range(1,EPOCHS), error[1:])
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
export_plot(plt,"training_error")


# Hidden representation
# =====================
# 
# - Red dots are the samples of the target pattern
# - Blue line is the model prediction in all the interval
# - The rest of the colors corresponds to the hidden activations

# In[10]:

plt.clf()
prediction = mlp.test(grid)
plt.scatter(X,Y,color='red')
plt.plot(grid,prediction)
z_hidden = mlp.feature_extraction(grid)
for i, z in enumerate(np.transpose(z_hidden)):
    plt.plot(grid, z)

plt.xlabel('input')
plt.ylabel('output')
export_plot(plt,"hidden_representation")


# In[11]:

mlp.print_architecture()

