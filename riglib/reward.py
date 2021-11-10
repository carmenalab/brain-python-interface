"""
Written by Pavi - Aug 2020 for reward system integration
Code for reward system used in Amy Orsborn lab
Functions list:
--------------
Class Basic -> reward(reward_time_s), test, calibrate, drain(drain_time_s)
"""

# import functions
from riglib.gpio import ArduinoGPIO
from multiprocessing import Process
from riglib import singleton
import time


class Basic(singleton.Singleton):

    __instance = None

    def __init__(self):
        super().__init__()
        com_port = '/dev/ttyACM0'  # specify the port, based on windows/Unix, can find it on IDE or terminal
        self.board = ArduinoGPIO(port=com_port)
        self.reward_pin = 12 # pin on the arduino which should be connected to the reward system
        self.off()

    def on(self):
        """Open the solenoid."""
        self.board.write(self.reward_pin, 1)  # send a high signal to open the solenoid

    def off(self):
        """Close the solenoid."""
        self.board.write(self.reward_pin, 0) # send a low signal to close the solenoid

    def calibrate(self):
        """
        reward.calibrate() checks if the flow rate of the reward system is consistent. Since our reward system is gravity based,
        the flowrate depends on the setup. Currently, the flow rate is 2.8 mL/s and it take ~ 72 seconds to drain 200 mL of fluid.
        #TODO: Check flowrate inside booth setup
        #TODO: Check flowrate for different fluid (Apple juice)
        """
        self.drain(72)  # it takes around 72 seconds to drain 200 ml of fluid - Flow rate: 2.8 mL/s
        print('Check the breaker for calibration. You should notice 200 ml of fluid')

    def drain(self, drain_time=200):  # call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
        self.on()
        time.sleep(drain_time)
        self.off()

    def async_drain(self, drain_time=200):
        """
        Calls drain() function in a separate process
        """
        p = Process(target=self.drain, args=((drain_time,)))
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
        traceback.print_exc(file=builtins.open('../log/reward.log', 'a'))