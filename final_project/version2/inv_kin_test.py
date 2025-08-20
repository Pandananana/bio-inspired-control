import time
from spatialmath import SE3
from fable import Fable
import numpy as np

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

# Generate a lawnmower pattern between x: -20 to +20, z: -20 to +60, at y=0
positions = []
x_range = [-20, 20]
z_start = 60
z_end = 80
z_step = 10  # adjust step size as needed

z_values = list(range(z_start, z_end + 1, z_step))
direction = 1  # 1 for left to right, -1 for right to left

for idx, z in enumerate(z_values):
    if direction == 1:
        x_values = range(x_range[0], x_range[1] + 1, 10)
    else:
        x_values = range(x_range[1], x_range[0] - 1, -10)
    for x in x_values:
        positions.append([x, 140, z])
    direction *= -1  # reverse direction for next row

for position in positions:
    try:
        print(f"Moving to: \n{position}")
        fable.setLaserPosition(position)
        time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        break


# fable.robot.teach()
