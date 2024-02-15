from .gpio import ArduinoGPIO
from multiprocessing import Process
from riglib import singleton
#from pyfirmata import Arduino, util
import traceback
import time
import os

log_path = os.path.join(os.path.dirname(__file__), '../log/reward.log')

class Basic(singleton.Singleton):

    __instance = None

    def __init__(self):
        super().__init__()
        self.board = ArduinoGPIO() # let the computer find the arduino. this won't work with more than one arduino!
        self.board.write(12,1)

    def dispense(self):  # call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
        self.board.write(12, 0)          #low
        time.sleep(0.02)
        self.board.write(12, 1)            #high

def open():
    try:
        reward = Basic.get_instance()
        return reward
    except:
        print("Reward system not found/ not active")
        import traceback
        import os
        import builtins
        traceback.print_exc()

import requests
class RemoteReward():

    def __init__(self):

        self.hostName = "192.168.0.200"
        self.serverPort = 8080

    def trigger(self):
        url = f"http://{self.hostName}:{self.serverPort}"
        try:
            requests.post(url, timeout=3)
        except:
            traceback.print_exc()
