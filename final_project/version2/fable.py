import subprocess
import numpy as np
from FableAPI.fable_init import api
from detect_ball import (
    locateV2,
    normalized_coordinates,
    camera_matrix,
)
import cv2
import matplotlib.pyplot as plt
from ikpy.chain import Chain
from spatialmath import SE3


class Fable:
    def __init__(self, robot_connected=True, camera_connected=True, camera_index=1):
        ## Fable API
        self.api = api
        self.robot_connected = robot_connected
        if robot_connected:
            self.api.setup(blocking=True)
            moduleids = self.api.discoverModules()
            self.module = moduleids[0] if moduleids else None
            self.getMotorAngles()
            print("Found modules: ", moduleids)
            print("Battery: ", self.getBattery())

        ## Camera
        self.camera_connected = camera_connected
        if camera_connected:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                print("Cannot open camera")
                exit()
        ## IKPY
        self.robot = Chain.from_urdf_file(
            "fable.urdf", active_links_mask=[False, True, True, True]
        )

        ## Storage
        self.ball_history = []
        self.angle_history = []

        # Add filter for Z values
        self.z_history = []
        self.z_filter_size = (
            10  # Number of previous values to keep for outlier detection
        )
        self.z_outlier_threshold = 2.0  # Standard deviations for outlier detection

    def setMotorAngles(self, tau_1, tau_2):
        # Exit if not connected
        if not self.robot_connected:
            return

        api.setPos(tau_1, tau_2, self.module)

    def getMotorAngles(self):
        # Exit if not connected
        if not self.robot_connected:
            return None

        angle0 = np.deg2rad(self.api.getPos(0, self.module))
        angle1 = np.deg2rad(self.api.getPos(1, self.module))

        self.angles = angle0, angle1

        return angle0, angle1

    def setLaserPosition(self, position):
        angles = self.inverseKinematics(position)
        self.setMotorAngles(np.rad2deg(angles[0]), np.rad2deg(angles[1]))

    def inverseKinematics(self, point):
        # Use the previous angles as the initial position
        if len(self.angle_history) == 0:
            previous_angles = [0, 0, 0, 0]
        else:
            previous_angles = self.angle_history[-1]

        # Inverse kinematics
        sol = self.robot.inverse_kinematics(
            point,
            initial_position=[
                0,
                previous_angles[0],
                previous_angles[1],
                previous_angles[2],
            ],
        )
        sol = sol[1:]
        self.angle_history.append(sol)
        return sol

    def forwardKinematics(self, angles):
        angles = [0, angles[0], angles[1], angles[2]]
        homotrans = self.robot.forward_kinematics(angles)
        return SE3(homotrans)

    def getBattery(self):
        if not self.robot_connected:
            return None
        return self.api.getBattery(self.module)

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
                return None, None

            x, y, r = locateV2(frame, hsv=True)
            x_norm, y_norm = normalized_coordinates(x, y)
            camera_x, camera_y, camera_z = self.ball_to_camera_coordinates(
                x_norm, y_norm, r
            )
            global_x, global_y, global_z = self.camera_to_global_coordinates(
                camera_x, camera_y, camera_z
            )
            self.ball_history.append((global_x, global_y, global_z))
            return (global_x, global_y, global_z), (x_norm, y_norm), frame

        except Exception as e:
            print(e)
            # print("Ball not found")
            return None, None, frame

    def ball_to_camera_coordinates(self, x_norm, y_norm, radius):
        """Convert normalized coordinates to camera coordinates"""

        if radius > 0:
            real_radius = 20  # mm
            f = camera_matrix[0, 0]  # Focal length from camera matrix
            Z = f * real_radius / radius  # Calculate depth based on radius
            Z = Z / 10  # Convert to cm

            # Apply outlier detection and filtering to Z
            Z_filtered = self._filter_z_outlier(Z)

            # x_norm and y_norm are already normalized coordinates from undistortPoints
            cam_x = x_norm * Z_filtered
            cam_y = y_norm * Z_filtered

            return cam_x, cam_y, Z_filtered
        else:
            raise ValueError("Invalid radius: must be greater than 0")

    def _filter_z_outlier(self, new_z):
        """Filter out outliers using statistical methods"""
        if len(self.z_history) == 0:
            # First measurement, just add it
            self.z_history.append(new_z)
            return new_z

        # Add new measurement
        self.z_history.append(new_z)

        # Keep only the last N measurements
        if len(self.z_history) > self.z_filter_size:
            self.z_history.pop(0)

        # Calculate statistics
        z_array = np.array(self.z_history)
        z_mean = np.mean(z_array)
        z_std = np.std(z_array)

        # Check if new value is an outlier
        if z_std > 0:  # Avoid division by zero
            z_score = abs(new_z - z_mean) / z_std

            if z_score > self.z_outlier_threshold:
                # Value is an outlier, use median of recent values instead
                z_filtered = np.median(z_array[:-1])  # Exclude the outlier
                print(
                    f"Z outlier detected: {new_z:.2f} (z-score: {z_score:.2f}), using filtered value: {z_filtered:.2f}"
                )
            else:
                # Value is not an outlier, use it
                z_filtered = new_z
        else:
            # No variation in data, use new value
            z_filtered = new_z

        return z_filtered

    def camera_to_global_coordinates(self, X, Y, Z):
        """
        Convert camera coordinates (X,Y,Z) to global coordinates.
        T_cam_ee : extrinsic transform from end-effector to camera (default identity if aligned)
        """

        # Extrinsic transform from end-effector to camera
        # Camera is 2cm above the end-effector, 7cm in front of the end-effector
        T_cam_ee = SE3(2, 0, 4.5) * SE3.RPY(0, 0, np.deg2rad(180))

        # Point in camera frame (convert from mm to cm)
        p_cam = SE3(X, Y, Z)

        # Get latest angles
        self.angles = self.getMotorAngles()

        # End-effector pose in global frame
        T_world_ee = self.forwardKinematics([self.angles[0], self.angles[1], 0])

        # Transform point to global frame
        p_global = T_world_ee * T_cam_ee * p_cam
        return p_global.t[:3]

    def error_point_to_middle_frame(self, X, Y):
        """
        Calculate the distance from (X, Y) to (0, 0)
        """
        return np.linalg.norm([X, Y])

    def plot_error_velocity_history(self, error):
        """
        Plot the error and velocity history of the ball in 3D space
        """
        if not self.ball_history or len(self.ball_history) < 2:
            print("No velocity to plot")
            return

        if error is None:
            print("No error to plot")
            return

        # Convert list of tuples to numpy array for indexing
        ball_history_array = np.array(self.ball_history)

        # Calculate velocities using np.diff (this reduces size by 1)
        velocities = np.diff(ball_history_array, axis=0)  # Shape: (n-1, 3)

        # Create 3 Plots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot X velocity
        axs[0].plot(velocities[:, 0], label="Velocity X")
        axs[0].plot(error[:], label="Error", color="red")
        axs[0].set_ylabel("Velocity X (cm/s)")
        axs[0].set_title("Velocity in X Direction Over Time")
        axs[0].legend()

        # Plot Y velocity
        axs[1].plot(velocities[:, 1], label="Velocity Y", color="orange")
        axs[1].plot(error[:], label="Error", color="red")
        axs[1].set_ylabel("Velocity Y (cm/s)")
        axs[1].set_title("Velocity in Y Direction Over Time")
        axs[1].legend()

        # Plot Z velocity
        axs[2].plot(velocities[:, 2], label="Velocity Z", color="green")
        axs[2].plot(error[:], label="Error", color="red")
        axs[2].set_ylabel("Velocity Z (cm/s)")
        axs[2].set_xlabel("Time Step")
        axs[2].set_title("Velocity in Z Direction Over Time")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

        # Get current git commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )

        # Save the plot in folder plots with commit hash in filename
        plt.savefig(f"plots/error_history_{commit_hash}.png")

    def plot_ball_history(self):
        """
        Make a 3D plot of the ball history with temporal coloring
        """
        # Get the ball history
        ball_history = self.ball_history

        # Convert list of tuples to numpy array for indexing
        if ball_history:
            ball_history_array = np.array(ball_history)
            num_points = len(ball_history_array)

            # Plot the ball history
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Create color gradient from blue (early) to red (late)
            colors = plt.cm.coolwarm(np.linspace(0, 1, num_points))

            # Plot all points except the last one with temporal coloring
            if num_points > 1:
                ax.scatter(
                    ball_history_array[:-1, 0],
                    ball_history_array[:-1, 1],
                    ball_history_array[:-1, 2],
                    c=colors[:-1],
                    s=50,  # size of regular points
                    alpha=0.7,
                )

            # Plot the last point with different marker and size to make it stand out
            ax.scatter(
                ball_history_array[-1, 0],
                ball_history_array[-1, 1],
                ball_history_array[-1, 2],
                c="red",
                s=200,  # larger size
                marker="*",  # star marker
                edgecolors="black",
                linewidth=2,
                alpha=1.0,
            )

            # Set labels and title
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Ball Position History (Temporal Progression)")

            # Get the current data ranges
            x_range = ball_history_array[:, 0].max() - ball_history_array[:, 0].min()
            y_range = ball_history_array[:, 1].max() - ball_history_array[:, 1].min()
            z_range = ball_history_array[:, 2].max() - ball_history_array[:, 2].min()

            # Find the maximum range to make all axes equal
            max_range = max(x_range, y_range, z_range)

            # Calculate centers for each axis
            x_center = (
                ball_history_array[:, 0].max() + ball_history_array[:, 0].min()
            ) / 2
            y_center = (
                ball_history_array[:, 1].max() + ball_history_array[:, 1].min()
            ) / 2
            z_center = (
                ball_history_array[:, 2].max() + ball_history_array[:, 2].min()
            ) / 2

            # Set equal limits for all axes
            ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
            ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
            ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

            # Add a colorbar to show temporal progression
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.coolwarm, norm=plt.Normalize(0, num_points - 1)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
            cbar.set_label("Time Step")

            plt.show()
        else:
            print("No ball history to plot")
