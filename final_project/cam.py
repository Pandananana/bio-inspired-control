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

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame")
            break

        # Display the frame
        cv2.imshow("Droidcam Feed", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Disconnected")


if __name__ == "__main__":
    show_droidcam_feed("http://10.110.60.67:4747/video")
