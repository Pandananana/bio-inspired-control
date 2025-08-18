from fable import Fable
from spatialmath import SE3
import numpy as np

fable = Fable(connection=False)

# Define the target point
point = SE3.Trans(-25, 20, 10)

# Try to solve inverse kinematics with error handling
solution = fable.inverseKinematics(point)
calculated_point = fable.forwardKinematics(solution)

## Logging
print("\n=== Results ===")
print("Target point:")
print(point)
print("Solution (joint angles):")
print(np.rad2deg(solution[:2]))
print(solution[2])
print(f"\nCalculated point from solution:")
print(calculated_point)

# Check if the solution is close to the target
print(f"Position error: {fable.getPositionError(point, calculated_point):.6f}")

fable.setLaserPosition(point)

# Plot the robot in the solution configuration
# fable.robot.plot(solution, backend="pyplot")
fable.robot.teach(solution)
