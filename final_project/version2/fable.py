import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from spatialmath import SE3


class Fable:
    def __init__(self, connection=True):
        ## Fable API
        self.api = api
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
                    a=9.5,
                    alpha=np.deg2rad(-90),
                    qlim=[np.deg2rad(-90), np.deg2rad(90)],
                ),
                rtb.PrismaticDH(a=0, theta=0, alpha=0, qlim=[0, 100]),
            ],
            name="Fable",
            base=SE3(0, 0, 22) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
        )

    def setMotorAngles(self, tau_1, tau_2):
        api.setPos(tau_1, tau_2, self.module)

        # Wait until both motors stop moving
        while api.getMoving(0, self.module) or api.getMoving(1, self.module):
            time.sleep(0.01)  # Small delay to avoid busy waiting

    def setLaserPosition(self, position):
        angles = self.inverseKinematics(position)
        self.setMotorAngles(np.rad2deg(angles[0]), np.rad2deg(angles[1]))

    def inverseKinematics(self, point):
        sol = self.robot.ik_LM(
            Tep=point,
            mask=[
                1,
                1,
                1,
                0,
                0,
                0,
            ],  # Only consider position (x,y,z), ignore orientation
            tol=1e-6,
            slimit=1000,
        )
        if sol[1]:
            return sol[0]
        else:
            print("No solution found")
            exit()

    def getBattery(self):
        return self.api.getBattery(self.module)
