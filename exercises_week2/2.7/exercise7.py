import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

## TODO: Paste your experiment code from exercise 2.6

## TODO: Change the code to the recurrent architecture
# You can update the cerebellum with: C = c.step(u, error)

## Initialize simulation
Ts = 1e-3
T_end = 5 # Period of one trial (T = 5 seconds)
n_steps = int(T_end/Ts) # in one trial
n_trials = 50 # Number of trials to run

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)
tau_m_vec = np.zeros(n_steps*n_trials)
#tau_cmac_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 150
Kv = 10

## TODO: Define parameters for periodic reference trajectory
A = np.pi  # Amplitude 
T = 5.0    # Period in seconds

## Adaptive filter cerebellum initialization
n_inputs = 1
n_outputs = 1
n_bases = 10
beta = 1e-6

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)
C = 0  # Initialize the recurrent term

## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    # Calculate current trial and time within trial
    current_trial = i // n_steps
    t_trial = (i % n_steps) * Ts  # Time within current trial (0 to T)
    
    ## Calculate the reference at this time step: θref = A sin(2π t / T)
    theta_ref = A * np.sin(2 * np.pi * t_trial / T)

    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controler
    error = (theta_ref - theta) + C
    tau_m = Kp * error + Kv* (-omega)
    
    C = c.step(tau_m, error)

    # Total control torque
    tau = tau_m # + tau_cmac
    
    # Iterate simulation dynamics
    plant.step(tau)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref
    tau_m_vec[i] = tau_m
    #tau_cmac_vec[i] = tau_cmac


## TODO: Plot results