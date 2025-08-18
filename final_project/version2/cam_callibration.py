import cv2 as cv
import numpy as np
import glob

# Chessboard size
chessboard_size = (9, 6)

# Square size in millimeters (or whatever unit you prefer)
square_size = 87.0  # Adjust this to your actual square size

# Termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
    * square_size
)

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = sorted(glob.glob("callibration_pictures/calibration_frame_*.png"))

for idx, fname in enumerate(images):
    print(f"Processing image {idx+1}/{len(images)}: {fname}")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
    else:
        print(f"Chessboard not found in {fname}")

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print(f"Overall RMS: {ret:.4f} pixels")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# Calculate per-image reprojection error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    print(f"{images[i]} RMS error: {error:.4f} pixels")
    total_error += error

print(f"Mean reprojection error: {total_error/len(objpoints):.4f} pixels")

"""
Results
------------------------------------------------------------------
Overall RMS: 0.5604 pixels

Camera matrix:
 [[843.52203357   0.         300.23349717]
 [  0.         842.38018005 259.67844506]
 [  0.           0.           1.        ]]
 
Distortion coefficients:
 [ 0.1579545  -0.70866363  0.00613847 -0.0072784   1.21241335]

 Mean reprojection error: 0.0593 pixels
 -----------------------------------------------------------------

"""


"""
Result from ASTA with big chessboard (6x9, 87mm square size)
Camera matrix:
 [[1.40068157e+03 0.00000000e+00 6.11754801e+02]
 [0.00000000e+00 1.39601724e+03 3.77394910e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Distortion coefficients:
 [ 1.55261819e-01 -7.87973206e-01 -2.48189023e-05 -4.35561861e-03
  1.04410200e+00]

  Mean reprojection error: 0.0891 pixels
"""
