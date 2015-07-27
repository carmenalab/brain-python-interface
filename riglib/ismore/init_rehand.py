import socket
import select
import time
from utils.constants import *

from riglib.ismore import settings

MAX_MSG_LEN = 200

REHAND_STATUS_LIST = [
    'READYTOGOHOME',
    'FAULT',
    'READYFORDONNING',
    'READY',
    'OPERATIONAL'
]

#nerea
sendSpeedCommand = False
watchdog = False

donning_position = np.array([int(p * rad_to_deg) for p in settings.donning_position.values])

# will be used to both send and recv
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# input address of ReHand application (address to send to)
rh_addr = settings.REHAND_UDP_SERVER_ADDR

sock.bind(settings.REHAND_UDP_CLIENT_ADDR)


def get_status():
    '''Sends a GetStatus command to the ReHand application and waits
    for a response packet. If either a) no response is received, or
    b) a different type of packet is received first (e.g., a packet
    containing feedback data), then this function just returns None.
    Otherwise, it returns the received status string.'''
    
    # send GetStatus command
    command = 'GetStatus ReHand\r'
    sock.sendto(command, rh_addr)

    # wait 1 sec to see if a response was received
    rlist, _, _ = select.select([sock], [], [], 1)
    if rlist:  # if rlist is not empty
        response = sock.recv(MAX_MSG_LEN)
        print "response", response

        # items is a list of strings (e.g., ['ReHand', 'READYFORDONNING']
        #   if the received packet was a response to the GetStatus command) 
        items = response.rstrip('\r').split(' ')

        dev_id = items[0]

        if (dev_id == 'ReHand') and (items[1] in REHAND_STATUS_LIST):
            status = items[1]
            return status
        else:
            return None
        
        print 'received status:', status
    else:
        return None

def recv_feedback():
    rlist, _, _ = select.select([sock], [], [], 1)
    if rlist:  # if rlist is not empty
        feedback = sock.recv(MAX_MSG_LEN)
        return feedback
    else:
        return None

def confirm_state(STATE):
    '''Repeatedly check ReHand state until it is in state STATE.'''
    while True:
        if get_status() == STATE:
            break
        else:
            time.sleep(1)
    print 'ReHand is now in state: ' + STATE


######## ACTUAL INITIALIZATION STEPS ########


# 1) Confirm that ReHand is in READYTOGOHOME state, then send GoHome command 
if get_status() != 'READYTOGOHOME':
    raise Exception('Make sure ReHand is in state READYTOGOHOME before running this script!')
print 'ReHand is in state: READYTOGOHOME'

raw_input("Please ensure a safe tilt angle of the forearm to avoid collisions between the hand module and the base cover and press Enter.")

command = 'GoHome ReHand\r'
sock.sendto(command, rh_addr)
time.sleep(5)


# 2) Confirm that ReHand is in READYFORDONNING state, then send GoDonning command
confirm_state('READYFORDONNING')

command = 'GoDonning ReHand %f %f %f %f\r' % tuple(donning_position)
sock.sendto(command, rh_addr)
time.sleep(5)


# 3) confirm that Rehand is in READY state, then send Go command
confirm_state('READY')

command = 'Go ReHand\r'
sock.sendto(command, rh_addr)
time.sleep(5)


print 'ReHand must now be in state: OPERATIONAL'
time.sleep(1)
print 'About to test receiving feedback (kill this process before running IsMore controller)'
time.sleep(1)

# 4) Check that we are receiving data from the hand and print it out
#    For 5 seconds, print out received feedback and estimated feedback rate in a loop
t_start = time.time()
recv_packet_count = 0

while time.time() - t_start < 3:
    feedback = recv_feedback()
    if feedback is not None:
        time.sleep(0.1)
        print feedback

        recv_packet_count += 1
        t_elapsed = time.time() - t_start
        print recv_packet_count / t_elapsed


# 5) disable motors so that the hand/forearm of the user can be
command = 'SystemDisable ReHand\r'
sock.sendto(command, rh_addr)
print "System disabled"

# 6) Put the patient's hand inside the ReHand
raw_input("Script is paused. Put the patient's hand inside the ReHand. Press Enter when finished to continue.")

print 'About to test receiving feedback (kill this process before running IsMore controller)'
time.sleep(1)

# 7) Initialization complete --> check that we are receiving data from the hand again, before starting any task recording
#    For 5 seconds, print out received feedback and estimated feedback rate in a loop
t_start = time.time()
recv_packet_count = 0

while time.time() - t_start < 3:
    feedback = recv_feedback()
    if feedback is not None:
        time.sleep(0.1)
        print feedback

        recv_packet_count += 1
        t_elapsed = time.time() - t_start
        print recv_packet_count / t_elapsed


print 'Exiting init_rehand.py script.'


# #when we get to the operational state we want to start sending speed commands constantly
# while True:
#     if get_status() == 'OPERATIONAL':
#         sendSpeedCommand = True
#         break
#     else:
#         time.sleep(1)
# print 'ReHand is in OPERATIONAL state'


# #changed position of this
# if settings.REHAND_WATCHDOG_ENABLED:
#     command = 'WatchDogEnable ReHand %d\r' % settings.WATCHDOG_TIMEOUT
#     sock.sendto(command, rh_addr)
#     print 'WatchDog enabled'


# #we send speed commands constantly
# while sendSpeedCommand == True:
#     command = 'SetSpeed ReHand 2 0 0 0\r' 
#     sock.sendto(command, rh_addr)
#    # print 'SetSpeed ReHand 0 0 0 0'
#     time.sleep(0.01)
    
#     #the watchdog is not enabled
#     #if watchdog == False:
#       #  command = 'WatchDogEnable ReHand %d\r' % settings.WATCHDOG_TIMEOUT
#      #   sock.sendto(command, rh_addr)
#      #   print 'WatchDog enabled'
#      #   watchdog = True

