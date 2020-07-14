from optitrack_interface import MotionData
import time

num_length  = 10
motion_data = MotionData(num_length)
motion_data.start()

while True:
    print(motion_data.get())
    time.sleep(1)