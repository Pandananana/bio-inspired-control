import numpy as np
import matplotlib.pyplot as plt

from robot import SingleLink
from cmac2 import CMAC

## Initialize simulation
Ts = 1e-2
T_end = 5 # Period of one trial (T = 5 seconds)
n_steps = int(T_end/Ts) # in one trial
n_trials = 50 # Number of trials to run

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)
tau_m_vec = np.zeros(n_steps*n_trials)
tau_cmac_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 150
Kv = 10

## TODO: Define parameters for periodic reference trajectory
A = np.pi  # Amplitude 
T = 5.0    # Period in seconds


## CMAC initialization
n_rfs = 11  # Number of radial basis functions
# Define input ranges for CMAC (theta_ref and theta)
xmin = [-np.pi, -np.pi]  # Minimum values for [theta_ref, theta]
xmax = [np.pi, np.pi]    # Maximum values for [theta_ref, theta]
beta = 1e-1             # Learning rate

cmac = CMAC(n_rfs, xmin, xmax, beta)



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
    error = (theta_ref - theta)
    tau_m = Kp * error + Kv* (-omega)

    # CMAC controller - predict feedforward torque
    x = [theta_ref, theta]  # Input to CMAC: [reference, actual position]
    tau_cmac = cmac.predict(x)
    
    # Total control torque
    tau = tau_m + tau_cmac
    
    # Iterate simulation dynamics
    plant.step(tau)
    
    # CMAC learning - use feedback torque as error signal
    # The idea is that if feedback torque is large, CMAC should learn to provide more feedforward
    cmac.learn(tau_m)

    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref
    tau_m_vec[i] = tau_m
    tau_cmac_vec[i] = tau_cmac



## Plotting
plt.figure(figsize=(12, 6))

# Plot all trials
plt.subplot(1, 2, 1)
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('All Trials')
plt.legend()
plt.grid(True)

# Add vertical lines to separate trials
for trial in range(1, n_trials):
    plt.axvline(x=trial*T_end, color='gray', linestyle=':', alpha=0.7)

# Plot last trial only
plt.subplot(1, 2, 2)
last_trial_start = (n_trials-1) * n_steps
t_last_trial = t_vec[last_trial_start:]
theta_last_trial = theta_vec[last_trial_start:]
theta_ref_last_trial = theta_ref_vec[last_trial_start:]

plt.plot(t_last_trial, theta_last_trial, label='theta')
plt.plot(t_last_trial, theta_ref_last_trial, '--', label='reference')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Last Trial')
plt.legend()
plt.grid(True)

plt.tight_layout()

## Plot trial error
error_vec = theta_ref_vec - theta_vec
trial_error = np.zeros(n_trials)
for trial in range(n_trials):
    trial_start = trial * n_steps
    trial_end = (trial + 1) * n_steps
    trial_error[trial] = np.sqrt(np.mean(error_vec[trial_start:trial_end]**2))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_trials+1), trial_error, 'o-')
plt.xlabel('Trial')
plt.ylabel('RMS Error (rad)')
plt.title('Trial Error Over Learning')
plt.grid(True)

# Plot CMAC contribution over last trial
plt.subplot(1, 2, 2)
last_trial_start = (n_trials-1) * n_steps
plt.plot(t_last_trial, tau_m_vec[last_trial_start:], label='Feedback torque')
plt.plot(t_last_trial, tau_cmac_vec[last_trial_start:], label='CMAC torque')
plt.plot(t_last_trial, tau_m_vec[last_trial_start:] + tau_cmac_vec[last_trial_start:], label='Total torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Control Torques - Last Trial')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

## Test generalization to different frequencies
def test_frequency(cmac, T_test, label, plant):
    """Test the learned CMAC on a different frequency"""
    # Reset plant state
    plant.theta = 0
    plant.omega = 0
    
    # Test parameters
    n_steps_test = int(T_test/Ts)
    t_test = np.array([Ts*i for i in range(n_steps_test)])
    
    theta_test = np.zeros(n_steps_test)
    theta_ref_test = np.zeros(n_steps_test)
    tau_m_test = np.zeros(n_steps_test)
    tau_cmac_test = np.zeros(n_steps_test)
    
    for i in range(n_steps_test):
        t_trial = i * Ts
        
        # Reference trajectory with new period
        theta_ref = A * np.sin(2 * np.pi * t_trial / T_test)
        
        # Measure
        theta = plant.theta
        omega = plant.omega
        
        # Feedback controller
        error = (theta_ref - theta)
        tau_m = Kp * error + Kv * (-omega)
        
        # CMAC controller (no learning, just prediction)
        x = [theta_ref, theta]
        tau_cmac = cmac.predict(x)
        
        # Total control torque
        tau = tau_m + tau_cmac
        
        # Iterate simulation dynamics
        plant.step(tau)
        
        # Store data
        theta_test[i] = plant.theta
        theta_ref_test[i] = theta_ref
        tau_m_test[i] = tau_m
        tau_cmac_test[i] = tau_cmac
    
    # Calculate RMS error
    error_test = theta_ref_test - theta_test
    rms_error = np.sqrt(np.mean(error_test**2))
    
    return t_test, theta_test, theta_ref_test, tau_m_test, tau_cmac_test, rms_error

# Test different frequencies
print("Testing generalization to different frequencies...")
print(f"Original training period T = {T} seconds")

# Store original plant state
original_theta = plant.theta
original_omega = plant.omega

# Test periods
test_periods = [2.5, 1.25, 10.0]  # Higher frequency (2.5s, 1.25s), Lower frequency (10s)
test_labels = ["Higher freq (T=2.5s)", "Much higher freq (T=1.25s)", "Lower freq (T=10s)"]

plt.figure(figsize=(12, 10))

# First pass: collect all data to determine y-axis limits
all_theta = []
all_theta_ref = []
all_errors = []

test_data = []
for T_test, label in zip(test_periods, test_labels):
    t_test, theta_test, theta_ref_test, tau_m_test, tau_cmac_test, rms_error = test_frequency(cmac, T_test, label, plant)
    test_data.append((t_test, theta_test, theta_ref_test, tau_m_test, tau_cmac_test, rms_error, label))
    
    all_theta.extend(theta_test)
    all_theta_ref.extend(theta_ref_test)
    all_errors.extend(theta_ref_test - theta_test)

# Determine consistent y-axis limits
theta_min = min(min(all_theta), min(all_theta_ref)) * 1.1
theta_max = max(max(all_theta), max(all_theta_ref)) * 1.1
error_min = min(all_errors) * 1.1
error_max = max(all_errors) * 1.1

# Second pass: plot with consistent axes
for idx, (t_test, theta_test, theta_ref_test, tau_m_test, tau_cmac_test, rms_error, label) in enumerate(test_data):
    # Plot tracking performance
    plt.subplot(3, 2, idx*2 + 1)
    plt.plot(t_test, theta_test, label='theta')
    plt.plot(t_test, theta_ref_test, '--', label='reference')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'{label}\nRMS Error: {rms_error:.4f} rad')
    plt.ylim(theta_min, theta_max)
    plt.legend()
    plt.grid(True)
    
    # Plot tracking error
    plt.subplot(3, 2, idx*2 + 2)
    error_test = theta_ref_test - theta_test
    plt.plot(t_test, error_test)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (rad)')
    plt.title('Tracking Error')
    plt.ylim(error_min, error_max)
    plt.grid(True)
    
    print(f"{label}: RMS Error = {rms_error:.4f} rad")

plt.tight_layout()
plt.show()

# Compare with feedback-only controller (no CMAC)
print("\nComparing with feedback-only controller...")

def test_feedback_only(T_test, plant):
    """Test feedback controller without CMAC"""
    plant.theta = 0
    plant.omega = 0
    
    n_steps_test = int(T_test/Ts)
    t_test = np.array([Ts*i for i in range(n_steps_test)])
    theta_test = np.zeros(n_steps_test)
    theta_ref_test = np.zeros(n_steps_test)
    
    for i in range(n_steps_test):
        t_trial = i * Ts
        theta_ref = A * np.sin(2 * np.pi * t_trial / T_test)
        
        theta = plant.theta
        omega = plant.omega
        
        error = (theta_ref - theta)
        tau = Kp * error + Kv * (-omega)  # Only feedback, no CMAC
        
        plant.step(tau)
        
        theta_test[i] = plant.theta
        theta_ref_test[i] = theta_ref
    
    error_test = theta_ref_test - theta_test
    rms_error = np.sqrt(np.mean(error_test**2))
    return rms_error

print("\nComparison: CMAC+Feedback vs Feedback-only")
print("Period\t\tCMAC+FB Error\tFB-only Error\tImprovement")
print("-" * 60)

for T_test, label in zip(test_periods, test_labels):
    # CMAC + Feedback error (already calculated)
    _, _, _, _, _, cmac_error = test_frequency(cmac, T_test, label, plant)
    
    # Feedback-only error
    fb_only_error = test_feedback_only(T_test, plant)
    
    improvement = ((fb_only_error - cmac_error) / fb_only_error) * 100
    print(f"{T_test}s\t\t{cmac_error:.4f}\t\t{fb_only_error:.4f}\t\t{improvement:.1f}%")
