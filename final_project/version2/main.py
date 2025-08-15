import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from fable import Fable
import roboticstoolbox as rtb
from spatialmath import SE3

fable = Fable()

tau_1 = 0
tau_2 = 0

fable.moveToPosition(tau_1, tau_2)


robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0, a=5, alpha=np.pi / 2, qlim=[-np.pi / 2, np.pi / 2]),
        rtb.RevoluteDH(d=0, a=20, alpha=0, qlim=[-np.pi / 2, np.pi / 2]),
    ],
    name="Fable",
    base=SE3(0, 0, 22) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
)

Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)  # solve IK
print(sol)

q_pickup = [np.deg2rad(tau_1), np.deg2rad(tau_2)]
print(sol[0])
print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

robot.plot(q_pickup, backend="pyplot")

# Wait for q key input
while True:
    time.sleep(60)
