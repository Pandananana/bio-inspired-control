import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from fable import Fable
import roboticstoolbox as rtb
from spatialmath import SE3

fable = Fable()

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0, a=5, alpha=np.pi / 2, qlim=[-np.pi / 2, np.pi / 2]),
        rtb.RevoluteDH(d=0, a=9.5, alpha=np.deg2rad(-90), qlim=[-np.pi / 2, np.pi / 2]),
        rtb.PrismaticDH(a=0, theta=0, alpha=0),
    ],
    name="Fable",
    base=SE3(0, 0, 22) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
)

point = SE3.Trans(0, -10, 55)
sol = robot.ik_LM(point)  # solve IK

q_pickup = sol[0]

fable.moveToPosition(np.rad2deg(q_pickup[0]), np.rad2deg(q_pickup[1]))
print("Solution: ", sol[0])
print(
    "FK: ", robot.fkine(q_pickup)
)  # FK shows that desired end-effector pose was achieved

robot.plot(q_pickup, backend="pyplot")

# Wait for q key input
while True:
    time.sleep(60)
