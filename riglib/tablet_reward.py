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
        self.reward_pin = 12 # pin on the arduino which should be connected to the reward system
        self.board.write(12,1)
        #self.on()

    # def calibrate(self):
    #     """
    #     reward.calibrate() checks if the flow rate of the reward system is consistent. Since our reward system is gravity based,
    #     the flowrate depends on the setup. Currently, the flow rate is 2.8 mL/s and it take ~ 72 seconds to drain 200 mL of fluid.
    #     #TODO: Check flowrate inside booth setup
    #     #TODO: Check flowrate for different fluid (Apple juice)
    #     """
    #     self.drain(72)  # it takes around 72 seconds to drain 200 ml of fluid - Flow rate: 2.8 mL/s
    #     print('Check the breaker for calibration. You should notice 200 ml of fluid')

    def dispense(self):  # call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
        self.board.write(12, 0)          #low
        time.sleep(0.02)
        self.board.write(12, 1)            #high

    def async_dispense(self, dispense_time=40):
        """
        Calls drain() function in a separate process
        """
        p = Process(target=self.dispense, args=((dispense_time,)))
        p.start()

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