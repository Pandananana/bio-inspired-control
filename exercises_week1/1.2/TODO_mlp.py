import numpy as np

from TODO_perceptron import Perceptron, SigmoidActivation, LinearActivation
from activation import ActivationFunction
import matplotlib.pyplot as plt

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
      n = len(inputs)

      # Initialize accumulators
      dw1 = np.zeros_like(self.l1.w)
      dw3 = np.zeros_like(self.l_out.w)
      # Loop over training examples
      for i in range(n):
         x = inputs[i]
         t = outputs[i]

         # Forward pass
         a1 = self.l1.activation(x)
         o1 = self.l1.output(a1)
         a3 = self.l_out.activation(o1)
         y = self.l_out.output(a3)

         # Backpropagation
         delta_out = self.l_out.gradient(a3) * (y - t)
         delta1 = self.l1.gradient(a1) * np.dot(self.l_out.w[1:], delta_out)

         # Add weight change contributions to temporary array
         o0 = np.insert(x, 0, 1)
         o1 = np.insert(o1, 0, 1)

         dw1 += delta1.reshape(-1,1).dot(o0.reshape(1,-1)).T
         dw3 += delta_out.reshape(-1,1).dot(o1.reshape(1,-1)).T

      # Update weights
      self.l1.update_weights(-self.alpha * dw1 / n)
      self.l_out.update_weights(-self.alpha * dw3 / n)
      

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
   # XOR input and target data
   X = np.array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
   ])

   T = np.array([0, 1, 1, 0])

   # Initialize MLP with 2 hidden units and 1 output
   mlp = MLP(num_inputs=2, n_hidden_units=2, n_outputs=1, alpha=0.1)

   # Training loop
   epochs = 2000
   mse_list = []

   for epoch in range(epochs):
      mlp.train(X, T)
      err = calc_prediction_error(mlp, X, T)
      mse_list.append(err)

      if epoch % 200 == 0 or epoch == epochs - 1:
         print(f"Epoch {epoch}: MSE = {err:.6f}")

   # Final predictions after training
   print("\nFinal predictions after training:")
   for x, t in zip(X, T):
      y = mlp.predict(x)
      print(f"Input: {x}, Target: {t}, Prediction: {y[0]:.4f}")

   # Plot the MSE over epochs
   plt.figure(figsize=(8, 5))
   plt.plot(mse_list)
   plt.xlabel("Epoch")
   plt.ylabel("Mean Squared Error (MSE)")
   plt.yscale("log")
   plt.title("XOR Training Error over Epochs")
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()
   pass
