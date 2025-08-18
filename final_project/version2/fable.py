import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from spatialmath import SE3


class Fable:
    def __init__(self, connection=True):
        ## Fable API
        self.api = api
        self.connection = connection
        if connection:
            self.api.setup(blocking=True)
            moduleids = self.api.discoverModules()
            self.module = moduleids[0] if moduleids else None
            print("Found modules: ", moduleids)
            print("Battery: ", self.getBattery())

        ## Robotics Toolbox
        self.robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(
                    d=0,
                    a=5,
                    alpha=np.deg2rad(90),
                    qlim=[np.deg2rad(-90), np.deg2rad(90)],
                ),
                rtb.RevoluteDH(
                    d=0,
                    a=8.5, #9.5,
                    alpha=np.deg2rad(-90),
                    qlim=[np.deg2rad(-90), np.deg2rad(90)],
                ),
                rtb.PrismaticDH(a=0, theta=0, alpha=0, qlim=[0, 100]),
            ],
            name="Fable",
            base=SE3(0, 0, 23) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
        )
        print(self.robot.n)

    def setMotorAngles(self, tau_1, tau_2):
        # Exit if not connected
        if not self.connection:
            return

        api.setPos(tau_1, tau_2, self.module)

        # Wait until both motors stop moving
        while api.getMoving(0, self.module) or api.getMoving(1, self.module):
            time.sleep(0.01)  # Small delay to avoid busy waiting

    def setLaserPosition(self, position):
        angles = self.inverseKinematics(position)
        self.setMotorAngles(np.rad2deg(angles[0]), np.rad2deg(angles[1]))

    def inverseKinematics(self, point):
        sol = self.robot.ik_gn(
            Tep=point,
            we=[
                1,
                1,
                1,
                0,
                0,
                0,
            ],  # Only consider position (x,y,z), ignore orientation
            reject_jl=True,
        )
        if not sol[1]:
            print("No solution found")
        return sol[0]

    def forwardKinematics(self, angles):
        return self.robot.fkine(angles)

    def getBattery(self):
        if not self.connection:
            return None
        return self.api.getBattery(self.module)

    def getPositionError(self, point1, point2):
        return np.linalg.norm(point1.t[:3] - point2.t[:3])
    
def camera_to_global_coordinates(self, X, Y, Z, angles):
    """
    Convert camera coordinates (X,Y,Z) to global coordinates.
    T_cam_ee : extrinsic transform from end-effector to camera (default identity if aligned)
    """
    # Extrinsic transform from end-effector to camera
    T_cam_ee = SE3(0, 0, 2) * SE3.RPY(0, 0, 0)

    # Point in camera frame
    p_cam = SE3(X, Y, Z)

    # End-effector pose in global frame
    T_world_ee = self.forwardKinematics([angles[0], angles[1], 0])

    # Transform point to global frame
    p_global = T_world_ee * T_cam_ee * p_cam
    return p_global.t[:3]
