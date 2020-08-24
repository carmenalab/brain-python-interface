"""
Written by Pavi - Aug 2020 for reward system integration
Code for reward system used in Amy Orsborn lab
Functions list:
--------------
Class Basic -> reward(reward_time_s), test, calibrate, drain(drain_time_s)
"""

# import functions
import pyfirmata
import time


class Basic(object):
    # com_port = '/dev/ttyACM0'  # specify the port, based on windows/Unix, can find it on IDE or terminal
    # board = pyfirmata.Arduino(com_port)

    def __init__(self):
        com_port = '/dev/ttyACM0'  # specify the port, based on windows/Unix, can find it on IDE or terminal
        self.board = pyfirmata.Arduino(com_port)

    def reward(self, reward_time_s=0.2):
        """Open the solenoid for some length of time. This function does not run the loop infinitely"""
        self.board.digital[13].write(1)  # send a high signal to Pin 13 on the arduino which should be connected to the reward system
        time.sleep(reward_time_s)  # in second
        print('ON')
        self.board.digital[13].write(0)
        print('OFF')

    def test(self):
        while True:
            self.board.digital[13].write(1)
            time.sleep(1)  # in second
            print('ON')
            self.board.digital[13].write(0)
            time.sleep(2)  # in secondS
            print('OFF')

    def calibrate(self):
        """
        reward.calibrate() checks if the flow rate of the reward system is consistent. Since our reward system is gravity based,
        the flowrate depends on the setup. Currently, the flow rate is 2.8 mL/s and it take ~ 72 seconds to drain 200 mL of fluid.
        #TODO: Check flowrate inside booth setup
        #TODO: Check flowrate for different fluid (Apple juice)
        """
        self.board.digital[13].write(1)
        time.sleep(72)  # it takes around 72 seconds to drain 200 ml of fluid - Flow rate: 2.8 mL/s
        self.board.digital[13].write(0)
        print('Check the breaker for calibration. You should notice 200 ml of fluid')

    def drain(self, drain_time=200):  # call this function to drain the reward system
        """
        this function is called from the webserver in ajax.reward_drain
        """
        # if cmd == 'ON':
        self.board.digital[13].write(1)
        time.sleep(drain_time)
        #   cmd = 'OFF'
        # if cmd == 'OFF':
        self.board.digital[13].write(0)


def open():
    try:
        reward = Basic()
        return reward
    except:
        print("Reward system not found/ not active")
        import traceback
        import os
        import builtins
        traceback.print_exc(file=builtins.open(os.path.expanduser('~/code/bmi3d/log/reward.log'), 'w'))