import cv2


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

    setup_trackbars("LAB")

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            break

        #rotate the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Original shape (1280, 720, 3)
        # Resize the frame to fit the window
        frame = cv2.resize(frame, (360, 640))

        frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) # LAB
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values("LAB")
        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        # Display the frame
        cv2.imshow("Droidcam Feed", thresh)

        # Break loop on 'q' press
        if cv2.waitKey(1) == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Disconnected")

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

if __name__ == "__main__":
    show_droidcam_feed("http://172.20.10.11:4747/video")
