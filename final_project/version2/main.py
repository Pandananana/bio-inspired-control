import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from fable import Fable
import roboticstoolbox as rtb
from spatialmath import SE3

# fable = Fable()

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0, a=5, alpha=np.pi / 2, qlim=[-np.pi / 2, np.pi / 2]),
        rtb.RevoluteDH(d=0, a=10, alpha=0, qlim=[-np.pi / 2, np.pi / 2]),
    ],
    name="Fable",
)

Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)  # solve IK
print(sol)

q_pickup = sol[0]
print(robot.fkine(q_pickup))  # FK shows that desired end-effector pose was achieved

qt = rtb.jtraj(robot.q, q_pickup, 50)
robot.plot(qt.q, backend="pyplot")
time.sleep(5)
