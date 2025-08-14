import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api


# Initialize the robot module
def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs.
    print("Search for modules")
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print("Found modules: ", moduleids)
    api.setPos(0, 0, module)
    api.sleep(0.5)

    return module


# Move the robot to the target pose
module = initialize_robot()

# Set move speed
speedX = 25
speedY = 25
api.setSpeed(speedX, speedY, module)

# Set accuracy
accurateX = "HIGH"
accurateY = "HIGH"
api.setAccurate(accurateX, accurateY, module)


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
q = [0, 0]  # joint angles in radians
T = robot.fkine(q)  # 4x4 transformation matrix
print("End effector pose:")
print(T)

# Inverse kinematics (what you want!)
q_solution = robot.ikine_LM(T)  # Levenberg-Marquardt solver
print("Joint angles:", q_solution.q)

api.setPos(q_solution.q[0], q_solution.q[1], module)
