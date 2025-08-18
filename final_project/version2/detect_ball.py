import cv2
import numpy as np

# Camera data (from ASTA with big chessboard 6x9, 87mm square size)
camera_matrix = np.array(
    [
        [1.40068157e03, 0.00000000e00, 6.11754801e02],
        [0.00000000e00, 1.39601724e03, 3.77394910e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
distortion_coefficients = np.array(
    [1.55261819e-01, -7.87973206e-01, -2.48189023e-05, -4.35561861e-03, 1.04410200e00]
)


def show_droidcam_feed(url: str):
    """Simple droidcam connection and video display"""
    print("Connecting to droidcam...")

    # Open video capture
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Cannot open droidcam connection")
        print(
            "Make sure droidcam is running on your phone and the IP address is correct"
        )
        return

    print("Connected! Press 'q' to quit")

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            break

        # Original shape (1280, 720, 3)
        # Resize the frame to fit the window
        # frame = cv2.resize(frame, (640, 360))

        x, y, r = locate(frame)

        if x is not None and y is not None and r is not None:
            x_norm, y_norm = normalized_coordinates(x, y)
            global_x, global_y, global_z = camera_coord(x_norm, y_norm, r)
            print(
                f"Global Coordinates: X={global_x:.2f}, Y={global_y:.2f}, Z={global_z:.2f}"
            )

        # Display the frame
        cv2.imshow("Droidcam Feed", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Disconnected")


def locate(img):
    frame_to_thresh = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # LAB
    thresh = cv2.inRange(frame_to_thresh, (63, 141, 76), (255, 255, 127))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape = img.shape

    if len(contours) != 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)

        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        center_upper_left_frame = (int(x), int(y))

        # Normalize center coordinates to be between -1 and 1
        center_x_norm = (x - (shape[1] / 2)) / (shape[1] / 2)
        center_y_norm = (y - (shape[0] / 2)) / (shape[0] / 2)

        print(f"Center: {center_x_norm:.2f}, {center_y_norm:.2f}")

        radius = int(radius)

        # Draw the circle and center
        cv2.circle(img, center_upper_left_frame, radius, (0, 0, 255), 2)
        cv2.circle(img, center_upper_left_frame, 5, (0, 255, 0), -1)

        return center_x_norm, center_y_norm, radius
    else:
        return ValueError("No ball found")


def normalized_coordinates(x, y):
    """Convert local coordinates to normalized image coordinates"""

    # OpenCV expects shape (N, 1, 2)
    image_point = np.array([[[x, y]]], dtype=np.float32)

    # Undistort the image point
    undistorted_point = cv2.undistortPoints(
        image_point, camera_matrix, distortion_coefficients
    )

    # normalized coordinates
    x_norm = undistorted_point[0][0][0]
    y_norm = undistorted_point[0][0][1]

    return x_norm, y_norm


def camera_coord(x_norm, y_norm, radius):
    """Convert normalized coordinates to global coordinates"""

    if radius > 0:
        real_radius = 20  # mm
        f = camera_matrix[0, 0]  # Focal length from camera matrix
        Z = f * real_radius / radius  # Calculate depth based on radius
        Z = Z / 10  # Convert to cm

        cam_x = x_norm * Z / 10  # Convert to cm
        cam_y = y_norm * Z / 10  # Convert to cm

        return cam_x, cam_y, Z
    else:
        raise ValueError("Invalid radius: must be greater than 0")


if __name__ == "__main__":
    show_droidcam_feed(1)
