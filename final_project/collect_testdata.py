import numpy as np
import camera_tools.camera_tools as ct
import cv2
import datetime
from FableAPI.fable_init import api
import time

# We have provided camera_tools as a stand-alone python file in ~/camera_tools/camera_tools.py
# The same functions are available in the Conda 'biocontrol' venv that is provided

cam = ct.prepare_camera()
print(cam.isOpened())
print(cam.read())

i = 0

def locate(img):
    frame_to_thresh = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # LAB
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

    return x, y

# Initialization of the camera. Wait for sensible stuff
def initialize_camera(cam):
    while True:
        frame = ct.capture_image(cam)

        x, _ = locate(frame)

        if x is not None:
            break     

def initialize_robot(module=None):
    api.setup(blocking=True)
    # Find all the robots and return their IDs
    print('Search for modules')
    moduleids = api.discoverModules()

    if module is None:
        module = moduleids[0]
    print('Found modules: ',moduleids)
    api.setPos(0,0, module)
    api.sleep(0.5)
    return module


initialize_camera(cam)
module = initialize_robot()

# Write DATA COLLECTION part - the following is a dummy code
# Remember to check the battery level and to calibrate the camera
# Some important steps are: 
# 1. Define an input/workspace for the robot; 
# 2. Collect robot data and target data
# 3. Save the data needed for the training

n_t1 = 10
n_t2 = 10

t1 = np.tile(np.linspace(-85, 86, n_t1), n_t2) # repeat the vector
t2 = np.repeat(np.linspace(0, 86, n_t2), n_t1) # repeat each element
thetas = np.stack((t1,t2))

num_datapoints = n_t1*n_t2

api.setPos(thetas[0,i], thetas[1,i], module)

class TestClass:
    def __init__(self, num_datapoints):
        self.i = 0
        self.num_datapoints = num_datapoints
        self.data = np.zeros( (num_datapoints, 4) )
        self.time_of_move = datetime.datetime.now()

    def go(self):
        if self.i >= num_datapoints:
            return True
        
        img = ct.capture_image(cam)
        x, y = locate(img)
        if (datetime.datetime.now() - self.time_of_move).total_seconds() > 2.0:
            if x is not None:
                print(x, y)
                tmeas1 = api.getPos(0,module)
                tmeas2 = api.getPos(1,module)
                self.data[self.i,:] = np.array([tmeas1, tmeas2, x, y])
                self.i += 1

                # set new pos
                if self.i != num_datapoints:
                    api.setPos(thetas[0,self.i], thetas[1,self.i], module)
                    self.time_of_move = datetime.datetime.now()
            else:
                print("Obj not found")
        
        # Display the frame
        cv2.imshow("Droidcam Feed", img)

        return False

test = TestClass(num_datapoints)

while True:
    k = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
    
    if k == ord(' '):  # Space key pressed
        if test.go():  # Returns True when all datapoints collected
            break
    elif k == 27:  # ESC key pressed
        print("Exiting early...")
        break
    
# Save collected data to CSV
np.savetxt("final_project_data.csv", test.data, delimiter=",", header="theta1,theta2,x,y", comments='')

print('Terminating')
api.terminate()

