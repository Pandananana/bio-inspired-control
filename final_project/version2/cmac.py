import numpy as np
import matplotlib.pyplot as plt
from fable import Fable

def GaussianBasisFunction(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (sigma**2))

class CMAC:
    def __init__(self, n_rfs, xmin, xmax, n_outputs=3, beta=1e-3):
        """Initialize the basis function parameters and output weights"""
        self.n_rfs = n_rfs
        self.n_outputs = n_outputs

        self.mu = np.zeros((2, self.n_rfs))
        self.sigma = np.zeros(2)
        crossval = 0.8  # has to be between 0 and 1 !

        for k in range(2):
            self.sigma[k] = (
                0.5
                / np.sqrt(-np.log(crossval))
                * (xmax[k] - xmin[k])
                / (self.n_rfs - 1)
            )  # RFs cross at phi = crossval
            self.mu[k] = np.linspace(xmin[k], xmax[k], self.n_rfs)

        # Now w has shape (n_outputs, n_rfs, n_rfs)
        self.w = np.random.normal(
            loc=0.0, scale=0.2, size=(self.n_outputs, self.n_rfs, self.n_rfs)
        )

        self.beta = beta

        self.B = None
        self.y = None
        self.w_history = []
        self.w_history.append(self.w.copy())

    def predict(self, x):
        """Predict yhat given x
        Saves activations `B` for later weight update
        """
        phi = np.zeros((2, self.n_rfs))
        for k in range(2):
            phi[k] = GaussianBasisFunction(
                x[k], self.mu[k], self.sigma[k]
            )  # for i in phi_ki at the same time

        self.B = np.zeros((self.n_rfs, self.n_rfs))
        for i in range(self.n_rfs):
            for j in range(self.n_rfs):
                self.B[i, j] = phi[0][i] * phi[1][j]

        # For each output, compute dot product
        yhat = np.zeros(self.n_outputs)
        for out in range(self.n_outputs):
            yhat[out] = np.dot(self.w[out].ravel(), self.B.ravel())

        return yhat

    def learn(self, e):
        """
        Update the weights using the covariance learning rule
        For all weights at once.
        e should be a vector of length n_outputs
        """
        for out in range(self.n_outputs):
            self.w[out] += self.beta * e[out] * self.B
        self.w_history.append(self.w.copy())
        return self.w
    
    def plot_weight_history(self):
        """Plot the history of all weights for all outputs"""
        w_hist = np.array(self.w_history)  # shape: (timesteps, n_outputs, n_rfs, n_rfs)
        timesteps = w_hist.shape[0]
        fig, axes = plt.subplots(self.n_outputs, 1, figsize=(12, 4 * self.n_outputs), sharex=True)

        if self.n_outputs == 1:
            axes = [axes]  # Ensure axes is iterable

        for out in range(self.n_outputs):
            ax = axes[out]
            for i in range(self.n_rfs):
                for j in range(self.n_rfs):
                    ax.plot(
                        range(timesteps),
                        w_hist[:, out, i, j],
                        label=f'w[{out},{i},{j}]',
                        alpha=0.6
                    )
            ax.set_title(f'Weight History for Output {out + 1}')
            ax.set_ylabel('Weight Value')
            ax.grid(True)
            # Optionally, comment out the next line if too many weights:
            # ax.legend(fontsize='small', ncol=4, bbox_to_anchor=(1.05, 1), loc='upper left')

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        plt.show()

#Example usage:
if __name__ == "__main__":
    cmac = CMAC(n_rfs=5, xmin=[-100, -100, -100], xmax=[100, 100, 100], n_outputs=2)
    fable = Fable(robot_connected=True, camera_connected=True, camera_index=1)

    xyz, _, _ = fable.detectBall() 
    yhat = cmac.predict(xyz[0], xyz[1])
    print("Predicted output:", yhat)

    # Simulate an error signal
    e = fable.angle_error(xyz)
    cmac.learn(e)

    # Plot the weight history
    cmac.plot_weight_history()