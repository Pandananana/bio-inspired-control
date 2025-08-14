import cv2
import numpy as np


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
        frame = cv2.resize(frame, (640, 360))

        locate(frame)

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
    thresh = cv2.inRange(frame_to_thresh, (152, 130, 91), (255, 170, 121))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)

        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the circle and center
        cv2.circle(img, center, radius, (0, 0, 255), 2)
        cv2.circle(img, center, 5, (0, 255, 0), -1)
    else:
        x = None
        y = None
        radius = None

    return x, y, radius


if __name__ == "__main__":
    show_droidcam_feed(0)
