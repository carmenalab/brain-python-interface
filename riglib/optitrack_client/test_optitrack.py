from riglib.optitrack_client.optitrack_interface import System
import time

num_length  = 10
motion_data = System()
motion_data.start()

while True:
    print(motion_data.get())
    time.sleep(0.05)