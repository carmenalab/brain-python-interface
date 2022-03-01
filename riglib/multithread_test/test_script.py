from .test_server import TestServer
from .test_server import TestServerMouse
from .test_client import TestClient
from .test_BMI3D_interface import MotionData
import time

#fire up the server
#test_server = TestServer()
test_server = TestServerMouse()
test_server.run()

num_length  = 10
motion_data = MotionData(num_length)
motion_data.start()

while True:
    print(motion_data.get())
    time.sleep(1)
    




#fire up the client
#t_client = TestClient()
#t_client.run()