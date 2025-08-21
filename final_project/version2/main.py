from fable import Fable
import cv2
import numpy as np
from cmac import CMAC
import matplotlib.pyplot as plt
from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
import subprocess

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

error = []

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:
        # print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
        fable.setLaserPosition([coords[0], coords[1], coords[2]])
        error.append([frame_coords[0], frame_coords[1]])

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Plot the error history
error = np.array(error)
plt.figure(figsize=(10, 5))
plt.plot(error, label="Error to middle frame")
plt.xlabel("Time Step")
plt.ylabel("Error Value")
plt.title("Error to Middle Frame Over Time")
plt.show()

# Get current git commit hash
commit_hash = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    .decode("utf-8")
    .strip()
)

# Save the plot in folder plots with commit hash in filename
plt.savefig(f"plots/error_history_{commit_hash}.png")

fable.plot_velocity_history()
