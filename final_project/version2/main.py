from cmac_xyz import CMAC3D
from fable import Fable
import cv2
import numpy as np
import matplotlib.pyplot as plt

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

errors = []

# while True:
#     coords, frame_coords, frame = fable.detectBall()

#     if coords and frame_coords:
#         # print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
#         # e = fable.angle_error([coords[0], coords[1], coords[2]])
#         fable.setLaserPosition([coords[0], coords[1], coords[2]], add_CMAC=True)
#         # fable.cmac.learn(e)
#         error.append(
#             fable.error_point_to_middle_frame(frame_coords[0], frame_coords[1])
#         )

#     fable.showFrame(frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Plot the weight history
# fable.cmac.plot_weight_history()
# # fable.plot_error_velocity_history(error)
# np.save("weights/cmac_weights.npy", fable.cmac.w)


## Lucas's code
enable_cmac = False
weighting_factor = 0.3
n_rfs = 10
lim = 100
load_weights = True

xmin = [-lim, -lim, -lim]
xmax = [lim, lim, lim]
cmac = CMAC3D(n_rfs, xmin, xmax, beta=1e-2)

if load_weights:
    try:
        cmac.w = np.load(f"weights/lucas_wf{weighting_factor}_nrf{n_rfs}_lim{lim}.npy")
    except:
        print("No weights found")

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:
        errors.append([frame_coords[0], frame_coords[1]])
        if enable_cmac:
            vel = fable.calculate_velocities()
            error = fable.positional_error(coords)
            if vel is not None and error is not None:
                new_pos = cmac.cmac_function(vel, error)
                delta_pos = (new_pos - coords) * weighting_factor
                position = coords + delta_pos
                fable.setLaserPosition(
                    [position[0], position[1], position[2]], add_CMAC=False
                )

        else:
            fable.setLaserPosition([coords[0], coords[1], coords[2]], add_CMAC=False)

    fable.showFrame(frame)

    # Close and save weights
    if cv2.waitKey(1) & 0xFF == ord("q"):
        np.save(f"weights/lucas_wf{weighting_factor}_nrf{n_rfs}_lim{lim}.npy", cmac.w)
        # cmac.plot_weight_history()
        break

# Improved error plot
plt.figure(figsize=(12, 6))

# Separate x and y errors
x_errors = [error[0] for error in errors]
y_errors = [error[1] for error in errors]

plt.subplot(2, 1, 1)
plt.plot(x_errors, label="X Error", color="tab:red", linewidth=2)
plt.title("X Error Over Time", fontsize=16)
plt.ylabel("X Error (pixels)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_errors, label="Y Error", color="tab:blue", linewidth=2)
plt.title("Y Error Over Time", fontsize=16)
plt.xlabel("Frame", fontsize=14)
plt.ylabel("Y Error (pixels)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
