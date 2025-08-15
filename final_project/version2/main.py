import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from fable import Fable
import roboticstoolbox as rtb
from spatialmath import SE3

fable = Fable(connection=False)


point = SE3.Trans(0, 36.5, 36.5)

solution = fable.inverseKinematics(point)
print("Solution:")
print(solution)

# fable.setLaserPosition(point)

fable.robot.plot(solution, backend="pyplot")

# Wait for q key input
while True:
    time.sleep(60)
