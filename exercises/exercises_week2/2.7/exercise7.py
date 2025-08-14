import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink


## Initialize simulation
Ts = 1e-3
T_end = 5  # Period of one trial (T = 5 seconds)
n_steps = int(T_end / Ts)  # in one trial
n_trials = 200  # Number of trials to run

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts * i for i in range(n_steps * n_trials)])

theta_vec = np.zeros(n_steps * n_trials)
theta_ref_vec = np.zeros(n_steps * n_trials)
tau_m_vec = np.zeros(n_steps * n_trials)
# tau_cmac_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 150
Kv = 10

## Define parameters for periodic reference trajectory
A = np.pi  # Amplitude
T = 5.0  # Period in seconds

## Adaptive filter cerebellum initialization
n_inputs = 1
n_outputs = 1
n_bases = 10
beta = 1e-6

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)
C = 0  # Initialize the recurrent term

## Simulation loop
for i in range(n_steps * n_trials):
    t = i * Ts
    # Calculate current trial and time within trial
    current_trial = i // n_steps
    t_trial = (i % n_steps) * Ts  # Time within current trial (0 to T)

    ## Calculate the reference at this time step: θref = A sin(2π t / T)
    theta_ref = A * np.sin(2 * np.pi * t_trial / T)

    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controler
    error = theta_ref - theta
    efb = error + C
    tau_m = Kp * efb + Kv * (-omega)

    C = c.step(tau_m, error)

    # Total control torque
    tau = tau_m  # + tau_cmac

    # Iterate simulation dynamics
    plant.step(tau)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref
    tau_m_vec[i] = tau_m
    # tau_cmac_vec[i] = tau_cmac

    # Print all values
    # print(f"t: {t}, theta: {theta}, theta_ref: {theta_ref}, tau_m: {tau_m}, C: {C}")


## Plotting
plt.figure(figsize=(18, 5))  # Wider for 1x3

# Plot all trials
plt.subplot(1, 3, 1)
plt.plot(t_vec, theta_vec, label="theta")
plt.plot(t_vec, theta_ref_vec, "--", label="reference")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("All Trials")
plt.legend()
plt.grid(True)
for trial in range(1, n_trials):
    plt.axvline(x=trial * T_end, color="gray", linestyle=":", alpha=0.7)

# Plot last trial only
plt.subplot(1, 3, 2)
last_trial_start = (n_trials - 1) * n_steps
t_last_trial = t_vec[last_trial_start:]
theta_last_trial = theta_vec[last_trial_start:]
theta_ref_last_trial = theta_ref_vec[last_trial_start:]
plt.plot(t_last_trial, theta_last_trial, label="theta")
plt.plot(t_last_trial, theta_ref_last_trial, "--", label="reference")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Last Trial")
plt.legend()
plt.grid(True)

# Plot trial error
error_vec = theta_ref_vec - theta_vec
trial_error = np.zeros(n_trials)
for trial in range(n_trials):
    trial_start = trial * n_steps
    trial_end = (trial + 1) * n_steps
    trial_error[trial] = np.sqrt(np.mean(error_vec[trial_start:trial_end] ** 2))

plt.subplot(1, 3, 3)
plt.plot(range(1, n_trials + 1), trial_error, "o-")
plt.xlabel("Trial")
plt.ylabel("RMS Error (rad)")
plt.title("Trial Error Over Learning")
plt.grid(True)

plt.tight_layout()
plt.show()
