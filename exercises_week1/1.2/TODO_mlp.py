import numpy as np

from TODO_perceptron import Perceptron, SigmoidActivation, LinearActivation
from activation import ActivationFunction

"""
HINT: Reuse your perceptron.py and activation.py files, and apply the functions directly.
"""


class Layer:
   def __init__(self, num_inputs, num_units, act_f):
      """ 
         Initialize the layer, creating `num_units` perceptrons with `num_inputs` each. 
      """
      # TODO Create the perceptrons required for the layer
      self.ps = []
      self.num_units = num_units
      for _ in range(num_units):
         p = Perceptron(num_inputs, act_f)
         self.ps.append(p)

   def activation(self, x):
      """ Returns the activation `a` of all perceptrons in the layer, given the input vector`x`. """
      return np.array([p.activation(x) for p in self.ps])

   def output(self, a):
      """ Returns the output `o` of all perceptrons in the layer, given the activation vector `a`. """
      return np.array([p.output(ai) for p, ai in zip(self.ps, a)])

   def predict(self, x):
      """ Returns the output `o` of all perceptrons in the layer, given the input vector `x`. """
      return np.array([p.predict(x) for p in self.ps])

   def gradient(self, a):
      """ Returns the gradient of the activation function for all perceptrons in the layer, given the activation vector `a`. """
      return np.array([p.gradient(ai) for p, ai in zip(self.ps, a)])

   def update_weights(self, dw):
      """ 
      Update the weights of all of the perceptrons in the layer, given the weight change of each.
      Input size: (n_inputs+1, n_units)
      """
      for i in range(self.num_units):
         self.ps[i].w += dw[:,i]

   @property
   def w(self):
      """
         Returns the weights of the neurons in the layer.
         Size: (n_inputs+1, n_units)
      """
      return np.array([p.w for p in self.ps]).T

   def import_weights(self, w):
      """ 
         Import the weights of all of the perceptrons in the layer.
         Input size: (n_inputs+1, n_units)
      """
      for i in range(self.num_units):
         self.ps[i].w = w[:,i]


class MLP:
   """ 
      Multi-layer perceptron class

   Parameters
   ----------
   n_inputs : int
      Number of inputs
   n_hidden_units : int
      Number of units in the hidden layer
   n_outputs : int
      Number of outputs
   alpha : float
      Learning rate used for gradient descent
   """
   def __init__(self, num_inputs, n_hidden_units, n_outputs, alpha=1e-3):
      self.num_inputs = num_inputs
      self.n_hidden_units = n_hidden_units
      self.n_outputs = n_outputs

      self.alpha = alpha

      # TODO: Define a hidden layer and the output layer
      self.l1 = Layer(num_inputs, n_hidden_units, SigmoidActivation)
      self.l_out = Layer(n_hidden_units, n_outputs, LinearActivation)

   def predict(self, x):
      """ 
      Forward pass prediction given the input x
      TODO: Write the function
      """
      y_l1 = self.l1.predict(x)
      y_l_out = self.l_out.predict(y_l1)
      return y_l_out

   def train(self, inputs, outputs):
      """
         Train the network

      Parameters
      ----------
      `x` : numpy array
         Inputs (size: n_examples, n_inputs)
      `t` : numpy array
         Targets (size: n_examples, n_outputs)

      TODO: Write the function to iterate through training examples and apply gradient descent to update the neuron weights
      """

      # Loop over training examples

         # Forward pass


         # Backpropagation


         # Add weight change contributions to temporary array
      o0 = np.insert(inp, 0, 1)
      o1 = np.insert(o1, 0, 1)

      dw1 += delta1.reshape(-1,1).dot(o0.reshape(1,-1))
      dw3 += delta_out.reshape(-1,1).dot(o1.reshape(1,-1))
         
         # Update weights
      
      return None # remove this line

   def export_weights(self):
      return [self.l1.w, self.l2.w]
   
   def import_weights(self, ws):
      if ws[0].shape == (self.l1.n_units, self.n_inputs+1) and ws[1].shape == (self.l2.n_units, self.l1.n_units+1):
         print("Importing weights..")
         self.l1.import_weights(ws[0])
         self.l2.import_weights(ws[1])
      else:
         print("Sizes do not match")


def calc_prediction_error(model, x, t):
   """ Calculate the average prediction error """
   # TODO Write the function
   n = len(t)
   error = np.zeros(n)

   for i in range(n):
      y = model.predict(x[i])
      error[i] = np.abs(y[0] - t[i])**2

   return np.sum(error) / n


if __name__ == "__main__":

   # TODO: Test new activation functions
   # Re-used from previous exercise

   # TODO: Test Layer class init
   x_test = np.array([[np.pi, 1]]).T
   l = Layer(2, 5, LinearActivation)
   print("\nLayer Test")
   print(l.predict(x_test))
   for p in l.ps:
      print(p.w)

   # TODO: Test MLP class init
   mlp = MLP(2, 5, 1)
   print("\nMLP Test")
   print(" - L1 shape: ", mlp.l1.w.shape)
   print(" - Lout shape: ", mlp.l_out.w.shape)
   print(" - Test predict: ", mlp.predict(x_test))

   # Test calc_prediction_error
   x_test = np.random.rand(10,2)
   t_test = np.random.rand(10)
   print(x_test)
   print(t_test)
   print(calc_prediction_error(mlp, x_test, t_test))

   # TODO: Training data

   # TODO: Initialization

   # TODO: Write a for loop to train the network for a number of iterations. Make plots.
   pass
