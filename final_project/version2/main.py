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
        rtb.RevoluteDH(
            d=0, a=5, alpha=np.deg2rad(90), qlim=[np.deg2rad(-90), np.deg2rad(90)]
        ),
        rtb.RevoluteDH(
            d=0, a=9.5, alpha=np.deg2rad(-90), qlim=[np.deg2rad(-90), np.deg2rad(90)]
        ),
        rtb.PrismaticDH(a=0, theta=0, alpha=0, qlim=[0, 100]),
    ],
    name="Fable",
    base=SE3(0, 0, 22) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
)

point = SE3.Trans(0, 36.5, 0)
print("Point:")
print(point)

sol = robot.ik_LM(
    Tep=point,
    mask=[1, 1, 1, 0, 0, 0],  # Only consider position (x,y,z), ignore orientation
    tol=1e-5,
)  # solve IK

success = sol[1]
if not success:
    print("No solution found")
    exit()

q_pickup = sol[0]
q_pickup = [np.deg2rad(-45), np.deg2rad(45), 50]

fable.moveToPosition(np.rad2deg(q_pickup[0]), np.rad2deg(q_pickup[1]))
print("Solution:")
print(sol[0])
print("FK:")
print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

robot.plot(q_pickup, backend="pyplot")

# Wait for q key input
while True:
    time.sleep(60)
