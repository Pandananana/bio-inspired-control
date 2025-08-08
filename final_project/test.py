import numpy as np
import torch
import torch_model
import cv2
import camera_tools as ct
from FableAPI.fable_init import api
from cam import locate

cam = ct.prepare_camera()
print(cam.isOpened())  # False
i = 0


# Initialize the camera first.. waits for it to detect the green block
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)
        x, _, _ = locate(frame)

        if x is not None:
            break


# Initialize the robot module
def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs.
    print("Search for modules")
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print("Found modules: ", moduleids)
    api.setPos(0, 0, module)
    api.sleep(0.5)

    return module


initialize_camera(cam)
module = initialize_robot()

# Set move speed
speedX = 25
speedY = 25
api.setSpeed(speedX, speedY, module)

# Set accuracy
accurateX = "HIGH"
accurateY = "HIGH"
api.setAccurate(accurateX, accurateY, module)

# TODO Load the trained model
model = torch_model.MLPNet(3, 16, 2)
model.load_state_dict(torch.load("trained_model.pth"))


while True:

    frame = ct.capture_image(cam)

    x, y, r = locate(frame)

    cv2.imshow("test", frame)

    with torch.no_grad():
        inp = torch.tensor([x, y, r]).float()
        outp = model(inp)
        t = outp.numpy()[0]
        print(t)
        api.setPos(t[0], t[1], module)

    # Break loop on 'q' press
    if cv2.waitKey(1) == ord("q"):
        break


print("Terminating")
api.terminate()
