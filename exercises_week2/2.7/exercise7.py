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
n_bases = 2
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


# Test generalization to different frequencies
def test_frequency(cerebellum, T_test, plant):
    """Test the learned cerebellum on a different frequency"""
    # Reset plant state
    plant.theta = 0
    plant.omega = 0

    n_steps_test = int(T_test / Ts)
    t_test = np.array([Ts * i for i in range(n_steps_test)])
    theta_test = np.zeros(n_steps_test)
    theta_ref_test = np.zeros(n_steps_test)
    tau_m_test = np.zeros(n_steps_test)
    C = 0  # recurrent term

    for i in range(n_steps_test):
        t_trial = i * Ts
        theta_ref = A * np.sin(2 * np.pi * t_trial / T_test)

        theta = plant.theta
        omega = plant.omega

        error = theta_ref - theta
        efb = error + C
        tau_m = Kp * efb + Kv * (-omega)

        C = cerebellum.step(tau_m, error)

        tau = tau_m  # + tau_cmac (not used here)

        plant.step(tau)

        theta_test[i] = plant.theta
        theta_ref_test[i] = theta_ref
        tau_m_test[i] = tau_m

    error_test = theta_ref_test - theta_test
    rms_error = np.sqrt(np.mean(error_test**2))
    return t_test, theta_test, theta_ref_test, tau_m_test, rms_error


print("Testing generalization to different frequencies...")
print(f"Original training period T = {T} seconds")

# Store original plant state
original_theta = plant.theta
original_omega = plant.omega

# Test periods
test_periods = [
    2.5,
    1.25,
    10.0,
]  # Higher frequency (2.5s, 1.25s), Lower frequency (10s)
test_labels = [
    "Higher freq (T=2.5s)",
    "Much higher freq (T=1.25s)",
    "Lower freq (T=10s)",
]

plt.figure(figsize=(12, 10))

all_theta = []
all_theta_ref = []
all_errors = []
test_data = []

for T_test, label in zip(test_periods, test_labels):
    t_test, theta_test, theta_ref_test, tau_m_test, rms_error = test_frequency(
        c, T_test, plant
    )
    test_data.append((t_test, theta_test, theta_ref_test, tau_m_test, rms_error, label))
    all_theta.extend(theta_test)
    all_theta_ref.extend(theta_ref_test)
    all_errors.extend(theta_ref_test - theta_test)

theta_min = min(min(all_theta), min(all_theta_ref)) * 1.1
theta_max = max(max(all_theta), max(all_theta_ref)) * 1.1
error_min = min(all_errors) * 1.1
error_max = max(all_errors) * 1.1

for idx, (
    t_test,
    theta_test,
    theta_ref_test,
    tau_m_test,
    rms_error,
    label,
) in enumerate(test_data):
    # Plot tracking performance
    plt.subplot(3, 2, idx * 2 + 1)
    plt.plot(t_test, theta_test, label="theta")
    plt.plot(t_test, theta_ref_test, "--", label="reference")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title(f"{label}\nRMS Error: {rms_error:.4f} rad")
    plt.ylim(theta_min, theta_max)
    plt.legend()
    plt.grid(True)

    # Plot tracking error
    plt.subplot(3, 2, idx * 2 + 2)
    error_test = theta_ref_test - theta_test
    plt.plot(t_test, error_test)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (rad)")
    plt.title("Tracking Error")
    plt.ylim(error_min, error_max)
    plt.grid(True)

    print(f"{label}: RMS Error = {rms_error:.4f} rad")

plt.tight_layout()
plt.show()
