from riglib.gpio import ArduinoGPIO
from multiprocessing import Process
from riglib import singleton
#from pyfirmata import Arduino, util
import time

class Basic(singleton.Singleton):

    __instance = None

    def __init__(self):
        super().__init__()
        com_port = 'COM4'  # specify the port, based on windows/Unix, can find it on IDE or terminal
        self.board = ArduinoGPIO(port=com_port)
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

        self.hostName = "localhost"
        self.serverPort = 8080

    def trigger(self):
        url = f"http://{self.hostName}:{self.serverPort}"
        requests.post(url)