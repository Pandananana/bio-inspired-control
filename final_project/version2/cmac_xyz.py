import numpy as np

def GaussianBasisFunction(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (sigma ** 2))

class CMAC3D:
    def __init__(self, n_rfs, xmin, xmax, beta):
        self.n_rfs = n_rfs
        self.beta = beta

        # Receptive field centers and widths for 3D input
        self.mu = np.zeros((3, n_rfs))
        self.sigma = np.zeros(3)
        crossval = 0.8
        for k in range(3):
            self.sigma[k] = 0.5 / np.sqrt(-np.log(crossval)) * (xmax[k] - xmin[k]) / (n_rfs - 1)
            self.mu[k] = np.linspace(xmin[k], xmax[k], n_rfs)

        # Weights: shape (n_rfs, n_rfs, n_rfs, 3) → 3 outputs
        self.w = np.random.normal(0.0, 0.2, size=(n_rfs, n_rfs, n_rfs, 3))

    def predict(self, vel):
        # Compute activations for each RF
        phi = np.zeros((3, self.n_rfs))
        for k in range(3):
            phi[k] = GaussianBasisFunction(vel[k], self.mu[k], self.sigma[k])

        # Compute combined activation tensor
        B = phi[0][:, None, None] * phi[1][None, :, None] * phi[2][None, None, :]

        # Weighted sum over all RFs for each output
        yhat = np.tensordot(B, self.w, axes=([0,1,2],[0,1,2]))
        self.B = B  # store for learning
        return yhat

    def learn(self, error):
        # Update all weights at once
        self.w += self.beta * self.B[:, :, :, None] * error[None, None, None, :]

    # The function you want: takes 3 vectors and returns new positions
    def cmac_function(self, vel, pos_actual, pos_correct):
        """
        vel: 3D velocity vector
        pos_actual: current ball position
        pos_correct: reference/correct position
        returns: new 3D position for robot
        """
        error = pos_correct - pos_actual
        new_pos = self.predict(vel)
        self.learn(error)
        return new_pos

# Example usage
if __name__ == '__main__':
    n_rfs = 7
    xmin = [-1, -1, -1]
    xmax = [1, 1, 1]
    cmac = CMAC3D(n_rfs, xmin, xmax, beta=1e-2)

    vel = np.array([0.1, -0.2, 0.05])
    pos_actual = np.array([0.5, 0.5, 0.5])
    pos_correct = np.array([0.6, 0.4, 0.55])

    new_pos = cmac.cmac_function(vel, pos_actual, pos_correct)
    # Save weights to a file
    np.save("/weights/test_weights.npy", cmac.w)
    print("New robot position:", new_pos)
