from fable import Fable
from spatialmath import SE3
import numpy as np

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

# Define the target point
point = SE3.Trans(0, 40, 0)

fable.setLaserPosition(point)

fable.detectBall()
