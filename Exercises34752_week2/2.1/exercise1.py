import numpy as np
import matplotlib.pyplot as plt

## Initialization
# Length of simulation (time steps)
simlen = 30

# TODO define the time delay
delay = [0,1,2,3]

# Output
y = np.zeros((simlen, len(delay)))
# Target
target = 0.0

# Controller gain
K = 1

# Set first output
y[0, :] = 1



## Simulation
for t in range(simlen-1):
    for d in delay:
        # Compute output
        # TODO include the time delay

        if t - d < 0:
            u = K * (target - y[0, d])
        else:
            u = K * (target - y[t - d, d])
        y[t+1, d]=0.5*y[t,d] + 0.4*u # 1st order dynamics

## Plot
time = range(simlen)
for d in delay:
    plt.plot(time, y[:, d], label=f'delay {d}')
plt.xlabel('time step')
plt.ylabel('y')
plt.show()