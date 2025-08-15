import numpy as np
import cv2 as cv
import os

# Camera initialization
def initialize_camera():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    return cap

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (chessboard 9x6 inner corners)
square_size = 25.0  # Set your square size here (e.g., 25.0 mm)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Ensure save directory exists
save_dir = "callibration_pictures"
os.makedirs(save_dir, exist_ok=True)

cap = initialize_camera()
if cap is None:
    print("Failed to initialize camera")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_corners, corners = cv.findChessboardCorners(gray, (9, 6), None)

        if ret_corners:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Save the original frame (without corners drawn)
            img_filename = os.path.join(save_dir, f"calibration_frame_{len(objpoints)}.png")
            cv.imwrite(img_filename, frame)

            # For display, draw corners on a copy
            frame_corners = frame.copy()
            cv.drawChessboardCorners(frame_corners, (9, 6), corners2, ret_corners)
        else:
            frame_corners = frame

        cv.imshow('img', frame_corners)
        key = cv.waitKey(1)
        if key == 27:  # ESC to exit
            break

        # Wait a bit before capturing the next frame
        cv.waitKey(500)

    cap.release()
    cv.destroyAllWindows()

