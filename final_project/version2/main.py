import roboticstoolbox as rtb
import numpy as np

# Create robot from DH parameters
# Your DH table: Link 1: d=0, θ=0, a=5, α=90°
#                Link 2: d=0, θ=0, a=10, α=0°

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0, a=5, alpha=np.pi / 2),  # Link 1
        rtb.RevoluteDH(d=0, a=10, alpha=0),  # Link 2
    ],
    name="Fable",
)

# Forward kinematics
q = [0.5, 1.0]  # joint angles in radians
T = robot.fkine(q)  # 4x4 transformation matrix
print("End effector pose:", T)

# Inverse kinematics (what you want!)
target_pose = rtb.SE3.Trans(12, 8, 0)  # target position
q_solution = robot.ikine_LM(target_pose)  # Levenberg-Marquardt solver
print("Joint angles:", q_solution.q)
