import pyfirmata 
import time 
#not sure about the protocal
com_port = '/dev/ttyACM0'#specify whihch port, can find it on IDE
board = pyfirmata.Arduino(com_port) 
 
def test_reward_system():
    while True: 
        board.digital[13].write(1) 
        time.sleep(1) #in second
        print('ON')
        board.digital[13].write(0)
        time.sleep(2) #in secondS
        print('OFF')

def cailbrate_reward():
    board.digital[13].write(1) 
    time.sleep(72) # it takes around 72 seconds to drain 200 ml of fluid - Flow rate: 2.8 mL/s
    board.digital[13].write(0)
    print('Check the breaker for calibration. You should notice 200 ml of fluid')

user_input = input("Enter 1 - to test Arduino connection; 2 - to calibrate:")

if user_input == '1':
    print('Testing Reward System')
    test_reward_system()
elif user_input == '2':
    print('Calibrating Reward System')
    cailbrate_reward()
