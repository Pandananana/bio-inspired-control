from fable import Fable
from spatialmath import SE3
import numpy as np
import time
import cv2

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

while True:
    coords, frame = fable.detectBall()

    if coords is not None:
        print(f"X: {coords[0]}, Y: {coords[1]}, Z: {coords[2]}")
        fable.setLaserPosition(SE3(coords[0], coords[1], coords[2]))

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
