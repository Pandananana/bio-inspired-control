from fable import Fable
from spatialmath import SE3

fable = Fable(connection=True)

# Define the target point
point = SE3.Trans(0, 50, 0)

# Try to solve inverse kinematics with error handling
solution = fable.inverseKinematics(point)

# This solution = [0, 0, 50] should equal point = SE3.Trans(14.5, 0, 10)
calculated_point = fable.forwardKinematics(solution)

## Logging
print("\n=== Results ===")
print("Target point:")
print(point)
print("Solution (joint angles):")
print(solution)
print("Calculated point from solution:")
print(calculated_point)

# Check if the solution is close to the target

print(f"Position error: {fable.getPositionError(point, calculated_point):.6f}")

fable.setLaserPosition(point)

# Plot the robot in the solution configuration
# fable.robot.plot(solution, backend="pyplot")
fable.robot.teach(solution)
