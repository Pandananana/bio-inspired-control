from cmac_xyz import CMAC3D
from fable import Fable
import cv2
import numpy as np
import matplotlib.pyplot as plt

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

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
enable_cmac = True
weighting_factor = 0.2
n_rfs = 10
lim = 100

xmin = [-lim, -lim, -lim]
xmax = [lim, lim, lim]
cmac = CMAC3D(n_rfs, xmin, xmax, beta=1e-2)

try:
    cmac.w = np.load(f"weights/lucas_wf{weighting_factor}_nrf{n_rfs}_lim{lim}.npy")
except:
    print("No weights found")

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:

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
        cmac.plot_weight_history()
        break
