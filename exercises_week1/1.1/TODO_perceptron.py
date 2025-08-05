import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt


##########################################
"""
TODO:
For each activation function: Sign, Sigmoid, Linear, define the following:
1) Forward function
2) Gradient function

We have provided the function signature for you.

"""
##########################################

# Helper function to compute sigmoid
def sigmoid(x):
   return 1/(1 + np.exp(-x))

class SignActivation(ActivationFunction):
   """ 
         Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
   """
   def forward(self, x):
      """
         This is the output function.
         TODO: Define the correct return function, given input `x`
      """
      if x > 0:
         return 1
      else:
         return 0
      
   def gradient(self, x):
      """
            Function derivative.
            Define the correct return value (derivative), given input `x`
      """
      return 0
   

class SigmoidActivation(ActivationFunction):
   def forward(self, x):
      return sigmoid(x)
   def gradient(self, x):
      return sigmoid(x) * (1 - sigmoid(x))


class LinearActivation(ActivationFunction):
   def forward(self, x):
      return x
   def gradient(self, x):
      return 1

class ReLU(ActivationFunction):
   pass


class Perceptron:
   """ 
      Perceptron neuron model
      Parameters
      ----------
      n_inputs : int
         Number of inputs
      act_f : Subclass of `ActivationFunction`
         Activation function
   """
   def __init__(self, n_inputs, act_f):
      """
         Perceptron class initialization
         TODO: Write the code to initialize weights and save the given activation function
      """
      if not isinstance(act_f, type) or not issubclass(act_f, ActivationFunction):
         raise TypeError('act_f has to be a subclass of ActivationFunction (not a class instance).')
      # weights
      mean = 0
      std = 0.5
      size = n_inputs + 1
      self.w = np.random.normal(mean, std, size)
      # activation function
      self.f = act_f()

      if self.f is not None and not isinstance(self.f, ActivationFunction):
         raise TypeError("self.f should be a class instance.")

   def activation(self, x):
      """
         It computes the activation `a` given an input `x`
         TODO: Fill in the function to provide the correct output
         NB: Remember the bias
      """
      a = np.sum(np.dot(self.w[1:], x)) + self.w[0]
      return a

   def output(self, a):
      """
         It computes the neuron output `y`, given the activation `a`
         TODO: Fill in the function to provide the correct output
      """
      y = self.f.forward(a)
      return y

   def predict(self, x):
      """
         It computes the neuron output `y`, given the input `x`
         TODO: Fill in the function to provide the correct output
      """
      a = self.activation(x)
      y = self.output(a)
      if y < 0:
         return 0
      else:
         return 1

   def gradient(self, a):
      """
         It computes the gradient of the activation function, given the activation `a`
      """
      return self.f.gradient(a)

if __name__ == '__main__':
   data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   xdata = data[:,:2]
   ydata = data[:,2]
   print(data)
   print(xdata)
   print(ydata)
   
## TODO Test your activation function
a = SignActivation()
print("Sign Activation")
print(a.forward(2))
print(a.forward(0))

b = SigmoidActivation()
print("")
print("Sigmoid Activation")
print(b.forward(2))
print(b.forward(0))

c = LinearActivation()
print("")
print("Linear Activation")
print(c.forward(2))
print(c.forward(0))


## TODO Test perceptron initialization
p = Perceptron(2,SigmoidActivation)
print("")
print("Initial Weights")
print(p.w)

print("\nPredict")
print(p.predict(xdata[0,:]))

## TODO Learn the weights
r = 0.1 # learning rate
for epoch in range(10):
   for i in range(len(xdata)):
      t = ydata[i]
      y = p.predict(xdata[i,:])
      error = t - y
      print("Error: ", error)
      p.w = p.w + r * error * xdata[i,:].insert()


# ## calculate the error and update the weights
# print(p.w)
# ## TODO plot points and linear decision boundary

# plt.plot(xp,yp, 'k--')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.show()