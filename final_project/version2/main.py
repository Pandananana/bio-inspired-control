from fable import Fable
import cv2
from cmac import CMAC
import matplotlib.pyplot as plt
from adaptive_filter.cerebellum import AdaptiveFilterCerebellum

fable = Fable(robot_connected=True, camera_connected=True, camera_index=0)

while True:
    coords, frame_coords, frame = fable.detectBall()

    if coords and frame_coords:
        # print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
        fable.setLaserPosition([coords[0], coords[1], coords[2]])

    fable.showFrame(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# # Initialize the adaptive filter cerebellum
# Ts = 1e-3  # Time step
# n_inputs = 2
# n_outputs = 3
# n_bases = 10
# beta = 1e-6

# c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)
# C = [0, 0, 0]  # Initialize the recurrent term

# w = []

# while True:
#     loop_start = time.time()  # Start time of the loop

#     coords, frame_coords, frame = fable.detectBall()

#     if coords and frame_coords:
#         print(f"X: {coords[0]:.2f}, Y: {coords[1]:.2f}, Z: {coords[2]:.2f}")
#         fable.setLaserPosition([coords[0] + C[0], coords[1] + C[1], coords[2] + C[2]])

#         # Learn the adaptive filter cerebellum
#         x = np.array([fable.angles[0], fable.angles[1]])
#         error = fable.error_point_to_middle_frame(frame_coords[0], frame_coords[1])
#         C = c.step(x, error)

#         w.append(c.weights.copy())

#     fable.showFrame(frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     # Sleep to maintain loop period Ts
#     elapsed = time.time() - loop_start
#     if elapsed < Ts:
#         time.sleep(Ts - elapsed)

# fable.plot_ball_history()

# Plot the weights
# w = np.array(w)
# plt.figure(figsize=(10, 5))
# for i in range(w.shape[1]):
#     for j in range(w.shape[2]):
#         plt.plot(w[:, i, j], label=f"Theta {i}, Receptive Field {j}")
# plt.xlabel("Time Step")
# plt.ylabel("Weight Value")
# plt.title("CMAC Weights Over Time")
# plt.legend()
# plt.show()


# while True:
#     fable.getMotorAngles()
#     point = fable.forwardKinematics([fable.angles[0], fable.angles[1], 0])
#     print(f"Point: {point}")
#     time.sleep(0.1)
