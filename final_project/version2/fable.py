import time
import roboticstoolbox as rtb
import numpy as np
from FableAPI.fable_init import api
from spatialmath import SE3
from detect_ball import (
    locate,
    normalized_coordinates,
    camera_coord,
    distortion_coefficients,
    camera_matrix,
)
import cv2


class Fable:
    def __init__(self, robot_connected=True, camera_connected=True, camera_index=1):
        ## Fable API
        self.api = api
        self.robot_connected = robot_connected
        if robot_connected:
            self.api.setup(blocking=True)
            moduleids = self.api.discoverModules()
            self.module = moduleids[0] if moduleids else None
            self.angles = self.getMotorAngles()
            print("Found modules: ", moduleids)
            print("Battery: ", self.getBattery())

        ## Camera
        self.camera_connected = camera_connected
        if camera_connected:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print("Cannot open camera")
                exit()

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
                    a=8.5,  # 9.5,
                    alpha=np.deg2rad(-90),
                    qlim=[np.deg2rad(-90), np.deg2rad(90)],
                ),
                rtb.PrismaticDH(a=0, theta=0, alpha=0, qlim=[0, 100]),
            ],
            name="Fable",
            base=SE3(0, 0, 23) * SE3.RPY(0, -np.deg2rad(90), -np.deg2rad(90)),
        )

    def setMotorAngles(self, tau_1, tau_2):
        # Exit if not connected
        if not self.robot_connected:
            return

        api.setPos(tau_1, tau_2, self.module)

        # Wait until both motors stop moving
        while api.getMoving(0, self.module) or api.getMoving(1, self.module):
            time.sleep(0.01)  # Small delay to avoid busy waiting

        self.angles = self.getMotorAngles()

    def getMotorAngles(self):
        # Exit if not connected
        if not self.robot_connected:
            return None

        angle0 = self.api.getPos(0, self.module)
        angle1 = self.api.getPos(1, self.module)

        return np.deg2rad(angle0), np.deg2rad(angle1)

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
        if not self.robot_connected:
            return None
        return self.api.getBattery(self.module)

    def getPositionError(self, point1, point2):
        return np.linalg.norm(point1.t[:3] - point2.t[:3])

    def showFrame(self, frame):
        if not self.camera_connected or frame is None:
            return
        cv2.imshow("Camera", frame)

    def detectBall(self):
        try:
            if not self.camera_connected:
                return

            # Read frame
            ret, frame = self.camera.read()

            if not ret:
                return ValueError("Can't receive frame")

            x, y, r = locate(frame)
            # TODO: Undistort using the normalized coordinates function
            # x_norm, y_norm = normalized_coordinates(x, y)
            camera_x, camera_y, camera_z = camera_coord(x, y, r)
            global_x, global_y, global_z = self.camera_to_global_coordinates(
                camera_x, camera_y, camera_z
            )
            return (global_x, global_y, global_z), frame
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def camera_to_global_coordinates(self, X, Y, Z):
        """
        Convert camera coordinates (X,Y,Z) to global coordinates.
        T_cam_ee : extrinsic transform from end-effector to camera (default identity if aligned)
        """

        # Extrinsic transform from end-effector to camera
        # Camera is 2cm above the end-effector, 7cm in front of the end-effector
        T_cam_ee = SE3(2, 0, 7) * SE3.RPY(0, 0, np.deg2rad(90))

        # Point in camera frame (convert from mm to cm)
        p_cam = SE3(X, Y, Z)

        # Get latest angles
        self.angles = self.getMotorAngles()

        # End-effector pose in global frame
        T_world_ee = self.forwardKinematics([self.angles[0], self.angles[1], 0])

        # Transform point to global frame
        p_global = T_world_ee * T_cam_ee * p_cam
        return p_global.t[:3]
