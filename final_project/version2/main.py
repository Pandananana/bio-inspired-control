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

# Create robot with joint limits
robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(
            d=0, a=5, alpha=np.pi / 2, qlim=[-np.pi / 2, np.pi / 2]
        ),  # Link 1: -90° to 90°
        rtb.RevoluteDH(
            d=0, a=10, alpha=0, qlim=[-np.pi / 2, np.pi / 2]
        ),  # Link 2: -90° to 90°
    ],
    name="Fable",
)

# Define target XYZ position (you can modify these values)
target_x = -8.0  # X coordinate
target_y = 5.0  # Y coordinate
target_z = 0.0  # Z coordinate (assuming planar movement)

print(f"Target position: X={target_x}, Y={target_y}, Z={target_z}")

# Create target pose matrix (4x4 transformation matrix)
# We'll use identity rotation (don't care about orientation)
target_pose = np.eye(4)
target_pose[0, 3] = target_x  # X translation
target_pose[1, 3] = target_y  # Y translation
target_pose[2, 3] = target_z  # Z translation

print("Target pose matrix:")
print(target_pose)

# Inverse kinematics to find joint angles
q_solution = robot.ikine_LM(target_pose)  # Levenberg-Marquardt solver
print("Joint angles (degrees):", np.degrees(q_solution.q))

# Move robot to the calculated joint angles
api.setPos(np.degrees(q_solution.q[0]), np.degrees(q_solution.q[1]), module)
api.sleep(5)
