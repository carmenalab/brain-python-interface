import socket
import select
import time

from riglib.ismore import settings

MAX_MSG_LEN = 200

REHAND_STATUS_LIST = [
    'READYTOGOHOME',
    'FAULT',
    'READYFORDONNING',
    'READY',
    'OPERATIONAL'
]

# all values in degrees
donning_position = [  
    30,  # thumb
    30,  # index
    30,  # fing3
    85   # prono
]

watchdog_timeout = 10000  # ms


# will be used to both send and recv
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# input address of ReHand application (address to send to)
rh_addr = settings.rehand_udp_server

sock.bind(settings.rehand_udp_client)
print settings.rehand_udp_client


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



######## ACTUAL INITIALIZATION STEPS ########

# 1) confirm that ReHand is in READYTOGOHOME state, then send GoHome command
status = get_status() 
print 'status', status
if status != 'READYTOGOHOME':
    raise Exception('Make sure ReHand is in state READYTOGOHOME before running this script!')
print 'ReHand is in state: READYTOGOHOME'

command = 'GoHome ReHand\r'
sock.sendto(command, rh_addr)
time.sleep(5)


# 2) wait until ReHand is in READYFORDONNING state, then send GoDonning command
while True:
    if get_status() == 'READYFORDONNING':
        break
    else:
        time.sleep(1)
print 'ReHand is now in state: READYFORDONNING'

command = 'GoDonning ReHand %f %f %f %f\r' % tuple(donning_position)
sock.sendto(command, rh_addr)
time.sleep(5)


######################################################################
# with real patient, would want to pause here for the actual donning #
######################################################################


# 3) wait until Rehand is in READY state, then send Go command
while True:
    if get_status() == 'READY':
        break
    else:
        time.sleep(1)
print 'ReHand is now in state: READY'

command = 'SetMaxTorque ReHand 600 600 3000 2000\r'
#sock.sendto(command, rh_addr)
time.sleep(3)

command = 'Go ReHand\r'
sock.sendto(command, rh_addr)
print 'ReHand must now be in state: OPERATIONAL'


# enable "watchdog" functionality
command = 'WatchDogEnable ReHand %d\r' % watchdog_timeout
# sock.sendto(command, rh_addr)


# 4) print out received feedback and estimated feedback rate in a loop

t_start = time.time()
recv_packet_count = 0
while True:
    feedback = recv_feedback()
    if feedback is not None:
        print feedback

        recv_packet_count += 1
        t_elapsed = time.time() - t_start
        print recv_packet_count / t_elapsed

# # wait until ReHand is in OPERATIONAL state
# while True:
#     status = get_status()
#     print status
#     if status == 'OPERATIONAL':
#     # if get_status() == 'OPERATIONAL':
#         break
#     else:
#         time.sleep(1)

# print 'ReHand is now in state: OPERATIONAL'


# ## enable "watchdog" functionality
# #command = 'WatchDogEnable ArmAssist %d\r' % watchdog_timeout
# #sock.sendto(command, rh_addr)
