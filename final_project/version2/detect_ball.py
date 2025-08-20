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


def create_seg_sliders():
    """Create segmentation sliders for color detection"""

    def nothing(x):
        pass

    # Create a window for the sliders
    cv2.namedWindow("Segmentation Sliders", cv2.WINDOW_NORMAL)

    # Create trackbars for lower and upper HSV bounds
    cv2.createTrackbar("Lower L", "Segmentation Sliders", 120, 255, nothing)
    cv2.createTrackbar("Lower A", "Segmentation Sliders", 50, 255, nothing)
    cv2.createTrackbar("Lower B", "Segmentation Sliders", 50, 255, nothing)
    cv2.createTrackbar("Upper L", "Segmentation Sliders", 160, 255, nothing)
    cv2.createTrackbar("Upper A", "Segmentation Sliders", 255, 255, nothing)
    cv2.createTrackbar("Upper B", "Segmentation Sliders", 255, 255, nothing)


def get_seg_values():
    """Get current segmentation values from sliders"""
    lower_l = cv2.getTrackbarPos("Lower L", "Segmentation Sliders")
    lower_a = cv2.getTrackbarPos("Lower A", "Segmentation Sliders")
    lower_b = cv2.getTrackbarPos("Lower B", "Segmentation Sliders")
    upper_l = cv2.getTrackbarPos("Upper L", "Segmentation Sliders")
    upper_a = cv2.getTrackbarPos("Upper A", "Segmentation Sliders")
    upper_b = cv2.getTrackbarPos("Upper B", "Segmentation Sliders")

    return lower_l, lower_a, lower_b, upper_l, upper_a, upper_b


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

        try:
            x, y, r = locateV2(frame, hsv=True)
        except ValueError as e:
            x, y, r = None, None, None
            print(e)

        if x is not None and y is not None and r is not None:
            x_norm, y_norm = normalized_coordinates(x, y)
            global_x, global_y, global_z = camera_coord(x_norm, y_norm, r)
            print(
                f"Camera Coordinates: X={global_x:.2f}, Y={global_y:.2f}, Z={global_z:.2f}"
            )

        # Display the frame
        cv2.imshow("Droidcam Feed", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
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
        x, y = (int(x), int(y))

        radius = int(radius)

        # Draw the circle and center
        cv2.circle(img, (x, y), radius, (0, 0, 255), 2)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        return x, y, radius
    else:
        return ValueError("No ball found")


def locateV2(frame, hsv=False):
    """
    Detects a single purple ping pong ball in an image frame.

    Args:
        frame: Input image frame (BGR format)

    Returns:
        tuple: (x, y, radius) if ball detected, None if no ball found
               x, y are center coordinates, radius is the ball radius in pixels
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")

    if hsv:
        # Convert BGR to LAB for perceptual color separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define purple color range in HSV
        # Purple typically has hue around 120-160, with good saturation and value
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([160, 255, 255])

        # Create mask for purple colors
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
    else:
        # Convert BGR to LAB for perceptual color separation
        frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # LAB
        mask = cv2.inRange(frame_to_thresh, (63, 141, 76), (255, 255, 127))

    # Morphological operations to clean up the mask
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No ball found")

    # Filter contours based on area and circularity
    valid_contours = []
    min_area = 100  # Minimum area threshold
    max_area = 100000  # Maximum area threshold

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip if area is too small or too large
        if area < min_area or area > max_area:
            continue

        # Calculate circularity to ensure it's round
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Ping pong balls should be fairly circular (circularity close to 1)
        if circularity > 0.5:  # Adjust threshold as needed
            valid_contours.append((contour, area))

    if not valid_contours:
        raise ValueError("No ball found")

    # Sort by area and take the largest valid contour
    # This helps when multiple purple objects are present
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    best_contour = valid_contours[0][0]

    # Fit ellipse to contour
    if len(best_contour) < 5:  # fitEllipse needs at least 5 points
        raise ValueError("Contour too small for ellipse fit")

    ellipse = cv2.fitEllipse(best_contour)
    (x, y), (major_axis, minor_axis), angle = ellipse

    # Use the minor axis as the "apparent diameter" (more stable than circle)
    radius = minor_axis / 2.0

    # Sanity check
    if radius < 10 or radius > 300:
        raise ValueError("Invalid radius from ellipse")

    # Draw the ellipse
    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    return int(x), int(y), int(radius)



def normalized_coordinates(x, y):
    """Convert local coordinates to normalized image coordinates"""

    # OpenCV expects shape (N, 1, 2)
    image_point = np.array([[[x, y]]], dtype=np.float32)

    # Undistort the image point
    undistorted_point = cv2.undistortPoints(
        image_point, camera_matrix, distortion_coefficients
    )

    # Undistorted coordinates
    x_norm = undistorted_point[0][0][0]
    y_norm = undistorted_point[0][0][1]

    return x_norm, y_norm


if __name__ == "__main__":
    show_droidcam_feed(0)
