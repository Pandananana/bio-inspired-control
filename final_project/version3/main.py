from ikpy.chain import Chain
from ikpy.link import Link
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D


def print_position(position):
    print("Position:")
    print(f"X: {position[0][3]:.2f}, Y: {position[1][3]:.2f}, Z: {position[2][3]:.2f}")


def print_solution(solution):
    print("Solution:")
    print(f"J1: {solution[1]:.2f}, J2: {solution[2]:.2f}, P: {solution[3]:.2f}")


fable = Chain.from_urdf_file("fable.urdf", active_links_mask=[False, True, True, True])

position = fable.forward_kinematics([0, np.deg2rad(-90), np.deg2rad(20), 100])
print_position(position)

ax = matplotlib.pyplot.figure().add_subplot(111, projection="3d")

target_position = [position[0][3], position[1][3], position[2][3]]
solution = fable.inverse_kinematics(
    target_position, initial_position=[0, np.deg2rad(0), np.deg2rad(0), 5]
)
print_solution(solution)

# Verify: convert solution back to degrees and compare
print(
    f"Verification - J1: {np.rad2deg(solution[1]):.1f}°, J2: {np.rad2deg(solution[2]):.1f}°, P: {solution[3]:.1f}"
)

# fable.plot(solution, ax)
# matplotlib.pyplot.show()
