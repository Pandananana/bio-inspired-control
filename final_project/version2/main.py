from fable import Fable
import cv2
import numpy as np
from cmac import CMAC
import matplotlib.pyplot as plt
from adaptive_filter.cerebellum import AdaptiveFilterCerebellum

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

error = []

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:
        # print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
        fable.setLaserPosition([coords[0], coords[1], coords[2]])
        error.append(
            fable.error_point_to_middle_frame(frame_coords[0], frame_coords[1])
        )

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

fable.plot_error_velocity_history(error)
