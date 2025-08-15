import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from spatialmath import SE3


class Fable:
    def __init__(self):
        ## Fable API
        self.api = api
        self.api.setup(blocking=True)
        moduleids = self.api.discoverModules()
        self.module = moduleids[0] if moduleids else None
        print("Found modules: ", moduleids)
        print("Battery: ", self.getBattery())

        ## Robotics Toolbox
        self.robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d=0, a=5, alpha=np.pi / 2, qlim=[-np.pi / 2, np.pi / 2]),
                rtb.RevoluteDH(d=0, a=20, alpha=0, qlim=[-np.pi / 2, np.pi / 2]),
            ],
            name="Fable",
            base=SE3(0, 0, 22) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
        )

    def moveToPosition(self, tau_1, tau_2):
        api.setPos(tau_1, tau_2, self.module)

        # Wait until both motors stop moving
        while api.getMoving(0, self.module) or api.getMoving(1, self.module):
            time.sleep(0.01)  # Small delay to avoid busy waiting

    def getBattery(self):
        return self.api.getBattery(self.module)
