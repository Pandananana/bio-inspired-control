from fable import Fable
from spatialmath import SE3
import numpy as np
import time
import cv2
from cmac import CMAC
import matplotlib.pyplot as plt

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

## CMAC initialization
n_rfs = 10  # Number of radial basis functions

# Define input ranges for CMAC (theta1 and theta2)
xmin = [-np.pi, -np.pi]
xmax = [np.pi, np.pi]
beta = 1e-1  # Learning rate

cmac = CMAC(n_rfs, xmin, xmax, n_outputs=2, beta=beta)

w = []

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:
        print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
        # fable.setLaserPosition(SE3(coords[0], coords[1], coords[2]))

        # Input to CMAC
        x = np.array([fable.angles[0], fable.angles[1]])

        cmac.predict(x)

        # Learn the CMAC
        error = fable.error_point_to_middle_frame(frame_coords[0], frame_coords[1])
        cmac.learn(error)

        w.append(cmac.w.copy())

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

fable.plot_ball_history()

# Plot the weights
w = np.array(w)
plt.figure(figsize=(10, 5))
for i in range(w.shape[1]):
    for j in range(w.shape[2]):
        plt.plot(w[:, i, j], label=f"Theta {i}, Receptive Field {j}")
plt.xlabel("Time Step")
plt.ylabel("Weight Value")
plt.title("CMAC Weights Over Time")
plt.legend()
plt.show()


# while True:
#     fable.getMotorAngles()
#     point = fable.forwardKinematics([fable.angles[0], fable.angles[1], 0])
#     print(f"Point: {point}")
#     time.sleep(0.1)
