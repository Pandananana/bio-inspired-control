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
        #fable.setLaserPosition(SE3(coords[0], coords[1], coords[2]))
        print("Frame error:",fable.error_point_to_middle_frame(coords[0],coords[1]) ,"Prismatic error:",fable.error_point_to_prismatic_line(coords))

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# while True:
#     fable.getMotorAngles()
#     point = fable.forwardKinematics([fable.angles[0], fable.angles[1], 0])
#     print(f"Point: {point}")
#     time.sleep(0.1)
