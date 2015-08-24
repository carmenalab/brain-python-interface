#!/usr/bin/python
'''
Code for interacting with the positioner microcontroller
'''
import serial
import time
import struct
import numpy as np

dir_lut = dict(x={0:0, -1:0, 1:1}, 
    y={0:0, -1:0, 1:1}, 
    z={0:1, -1:1, 1:0}, # convention flipped for z-stage
) 

class Positioner(object):
    def __init__(self, dev='/dev/ttyACM1'):
        self.port = serial.Serial(dev, baudrate=115200)

    def _parse_resp(self, resp):
        resp = resp.rstrip()
        limits = map(int, resp[-6:])
        return limits

    def poll_limit_switches(self, N=100):
        while 1:
            time.sleep(0.1)
            self.port.write('\n')
            raw_resp = self.port.readline()
            print "limit switches", self._parse_resp(raw_resp)

    def read_limit_switches(self):
        self.port.write('\n')
        raw_resp = self.port.readline()
        return self._parse_resp(raw_resp)

    def wake_motors(self):
        self.port.write('w\n')
        self.port.readline()

    def sleep_motors(self):
        self.port.write('s\n')
        self.port.readline()

    def step_motors(self, step_x, step_y, step_z, dir_x, dir_y, dir_z):
        cmd_data = 0
        cmd_step_data = step_x | (step_y << 1) | (step_z << 2)
        cmd_dir_data = dir_x | (dir_y << 1) | (dir_z << 2)
        cmd_data = cmd_step_data | (cmd_dir_data << 4)
        cmd = 'm' + struct.pack('B', cmd_data) + '\n'
        #print cmd_data, cmd
        self.port.write(cmd)

    def move(self, n_steps_x, n_steps_y, n_steps_z):
        self.wake_motors()
        limits = self._parse_resp(self.port.readline())
        time.sleep(1)

        dir_x = dir_lut['x'][np.sign(n_steps_x)]
        dir_y = dir_lut['y'][np.sign(n_steps_y)]
        dir_z = dir_lut['z'][np.sign(n_steps_z)]

        n_steps_sent_x = 0
        n_steps_sent_y = 0
        n_steps_sent_z = 0

        k = 0
        while (abs(n_steps_x) > n_steps_sent_x) or (abs(n_steps_y) > n_steps_sent_y) or (abs(n_steps_z) > n_steps_sent_z):
            if k % 10 == 0: print k
            step_x = int(n_steps_sent_x < abs(n_steps_x))
            step_y = int(n_steps_sent_y < abs(n_steps_y))
            step_z = int(n_steps_sent_z < abs(n_steps_z))
            #print step_x, step_y, step_z, dir_x, dir_y, dir_z
            self.step_motors(step_x, step_y, step_z, dir_x, dir_y, dir_z)
            limits = self._parse_resp(self.port.readline())
            k += 1

            n_steps_sent_x += step_x
            n_steps_sent_y += step_y
            n_steps_sent_z += step_z
        
        self.sleep_motors()

    def old_move(self, n_steps_x, n_steps_y, n_steps_z):
        self.wake_motors()
        try:
            time.sleep(1)

            dir_x = dir_lut['x'][np.sign(n_steps_x)]
            dir_y = dir_lut['y'][np.sign(n_steps_y)]
            dir_z = dir_lut['z'][np.sign(n_steps_z)]

            n_steps_sent_x = 0
            n_steps_sent_y = 0
            n_steps_sent_z = 0

            k = 0
            while (abs(n_steps_x) > n_steps_sent_x) or (abs(n_steps_y) > n_steps_sent_y) or (abs(n_steps_z) > n_steps_sent_z):
                if k % 10 == 0: print k
                step_x = int(n_steps_sent_x < abs(n_steps_x))
                step_y = int(n_steps_sent_y < abs(n_steps_y))
                step_z = int(n_steps_sent_z < abs(n_steps_z))
                #print step_x, step_y, step_z, dir_x, dir_y, dir_z
                self.step_motors(step_x, step_y, step_z, dir_x, dir_y, dir_z)
                limits = self._parse_resp(self.port.readline())
                k += 1

                n_steps_sent_x += step_x
                n_steps_sent_y += step_y
                n_steps_sent_z += step_z
            
        except:
            import traceback
            traceback.print_exc()
        finally:
            self.sleep_motors()

    
    
    def move2(self):
        self.wake_motors()
        for k in range(200):
            self.step_motors(1,0,0,0,0,0)
            limits = self._parse_resp(self.port.readline())
        self.sleep_motors()

    def go_to_min(self, verbose=False):
        can_move = self.read_limit_switches()
        x_can_decrease = can_move[0]
        y_can_decrease = can_move[2]
        z_can_decrease = can_move[4]

        dir_x = dir_lut['x'][-1]
        dir_y = dir_lut['y'][-1]
        dir_z = dir_lut['z'][-1]

        n_steps_sent_x = 0
        n_steps_sent_y = 0
        n_steps_sent_z = 0
        if x_can_decrease or y_can_decrease or z_can_decrease:
            self.wake_motors()

        try:
            k = 0
            while x_can_decrease or y_can_decrease or z_can_decrease:
                step_x = int(x_can_decrease)
                step_y = int(y_can_decrease)
                step_z = int(z_can_decrease)
                if verbose:
                    print step_x, step_y, step_z

                self.step_motors(step_x, step_y, step_z, dir_x, dir_y, dir_z)
                can_move = self._parse_resp(self.port.readline())

                x_can_decrease = can_move[0]
                y_can_decrease = can_move[2]
                z_can_decrease = can_move[4]

                n_steps_sent_x += step_x
                n_steps_sent_y += step_y
                n_steps_sent_z += step_z
                k += 1
        finally:
            self.sleep_motors()

        return n_steps_sent_x, n_steps_sent_y, n_steps_sent_z

    def go_to_max(self, verbose=False):
        can_move = self.read_limit_switches()
        x_can_increase = can_move[1]
        y_can_increase = can_move[3]
        z_can_increase = can_move[5]

        dir_x = dir_lut['x'][1]
        dir_y = dir_lut['y'][1]
        dir_z = dir_lut['z'][1]

        n_steps_sent_x = 0
        n_steps_sent_y = 0
        n_steps_sent_z = 0
        if x_can_increase or y_can_increase or z_can_increase:
            self.wake_motors()

        try:
            k = 0
            while x_can_increase or y_can_increase or z_can_increase:
                step_x = int(x_can_increase)
                step_y = int(y_can_increase)
                step_z = int(z_can_increase)
                if verbose:
                    print step_x, step_y, step_z

                self.step_motors(step_x, step_y, step_z, dir_x, dir_y, dir_z)
                can_move = self._parse_resp(self.port.readline())

                x_can_increase = can_move[1]
                y_can_increase = can_move[3]
                z_can_increase = can_move[5]

                n_steps_sent_x += step_x
                n_steps_sent_y += step_y
                n_steps_sent_z += step_z
                k += 1
        finally:
            self.sleep_motors()

        return n_steps_sent_x, n_steps_sent_y, n_steps_sent_z

    def continuous_move(self, n_steps_x, n_steps_y, n_steps_z):
        self.wake_motors()
        msg = 'c' + struct.pack('>hhh', n_steps_x, n_steps_y, n_steps_z) + '\n'
        self.port.write(msg)
        movement_data = self.port.readline()
        limit_switch_data = self.port.readline()

        m = re.match(".*?: (\d+), (\d+), (\d+)", ex)
        n_steps_actuated = map(int, [m.group(x) for x in [1,2,3]])
        self.sleep_motors()

        return n_steps_actuated

    def start_continuous_move(self, n_steps_x, n_steps_y, n_steps_z):
        '''
        Same as 'continuous_move', but without blocking for a response/movement to finish before the function returns
        '''
        self.wake_motors()
        msg = 'c' + struct.pack('>hhh', n_steps_x, n_steps_y, n_steps_z) + '\n'
        self.port.write(msg)

        self.motor_dir = np.array([np.sign(n_steps_x), np.sign(n_steps_y), np.sign(n_steps_z)])

    def end_continuous_move(self):
        '''
        Cleanup part of 'continuous_move' after 'start_continuous_move' has been called
        '''
        movement_data = self.port.readline()
        limit_switch_data = self.port.readline()

        m = re.match(".*?: (\d+), (\d+), (\d+)", ex)
        n_steps_actuated = map(int, [m.group(x) for x in [1,2,3]])
        self.sleep_motors()

        return n_steps_actuated

    def calibrate(self, n_runs):
        '''
        Repeatedly go from min to max and back so the number of steps can be counted
        '''
        n_steps_min_to_max = [None]*n_runs
        n_steps_max_to_min = [None]*n_runs

        self.go_to_min()
        time.sleep(1)

        for k in range(n_runs):
            n_steps_min_to_max[k] = self.go_to_max()
            time.sleep(2)
            n_steps_max_to_min[k] = self.go_to_min()
            print "min to max"
            print n_steps_min_to_max
            print "max to min"
            print n_steps_max_to_min
            time.sleep(2)
            
        return n_steps_min_to_max, n_steps_max_to_min

    def data_available(self):
        return self.port.inWaiting()

from riglib.experiment import Experiment, FSMTable, StateTransitions
class PositionerTaskController(Experiment):
    '''
    Interface between the positioner and the task interface. The positioner should run asynchronously
    so that the task event loop does not have to wait for a serial port response from the microcontroller.
    '''

    status = FSMTable(
        go_to_origin = StateTransitions(microcontroller_done='wait')
        wait = StateTransitions(start_trial='move_target'),
        move_target = StateTransitions(microcontroller_done='reach'),
        reach = StateTransitions(time_expired='reward', new_target_set_remotely='move_target'),
        reward = StateTransitions(time_expired='wait'),
    )

    sequence_generators = ['random_target']

    @staticmethod
    def random_target(length=100):
        raise NotImplementedError

    def __init__(self, *args, **kwargs, positioner_dev='/dev/ttyACM1', 
        x_cm_per_rev=12, y_cm_per_rev=12, z_cm_per_rev=7.6, x_step_size=1./4, y_step_size=1./4, z_step_size=1./4):
        '''
        Constructor for PositionerTaskController

        Parameters
        ----------
        # x_len : float
        #     measured distance the positioner can travel in the x-dimension
        # y_len : float
        #     measured distance the positioner can travel in the y-dimension
        # z_len : float
        #     measured distance the positioner can travel in the z-dimension
        dev : str, optional, default=/dev/ttyACM1
            Serial port to use to communicate with Arduino controller
        x_cm_per_rev : int, optional, default=12
            Number of cm traveled for one full revolution of the stepper motors in the x-dimension
        y_cm_per_rev : int, optional, default=12
            Number of cm traveled for one full revolution of the stepper motors in the y-dimension
        z_cm_per_rev : float, optional, default=7.6
            Number of cm traveled for one full revolution of the stepper motors in the z-dimension
        x_step_size : float, optional, default=0.25
            Microstepping mode in the x-dimension
        y_step_size : float, optional, default=0.25
            Microstepping mode in the y-dimension
        z_step_size : float, optional, default=0.25
            Microstepping mode in the z-dimension
    
        Returns
        -------
        PositionerTaskController instance
        '''
        self.loc = np.ones(3) * np.nan # position of the target relative to origin is unknown until the origin limit switches are hit
        self.pos_uctrl_iface = Positioner(dev=positioner_dev)
        self.pos_uctrl_iface.sleep_motors()

        self.steps_from_origin = np.ones(3) * np.nan # cumulative number of steps taken from the origin. Unknown until the origin limit switches are hit.
        self.step_size = np.array([x_step_size, y_step_size, z_step_size], dtype=np.float64)
        self.cm_per_rev = np.array([x_cm_per_rev, y_cm_per_rev, z_cm_per_rev], dtype=np.float64)

        self.full_steps_per_rev = 200.

        super(PositionerTaskController, self).__init__(self, *args, **kwargs)

    def init(self):
        super(PositionerTaskController, self).init()

        # open an rx_socket for reading new commands 
        import socket
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx_sock.bind(('localhost', 60005))

    def terminate(self):
        # close the rx socket used for reading remote commands
        super(PositionerTaskController, self).terminate()
        self.rx_sock.close()

    ##### Helper functions #####
    def _calc_steps_to_pos(self, target_pos):
        displ_cm = target_pos - self.loc

        # compute the number of steps needed to travel the desired displacement
        displ_rev = displ_cm / self.cm_per_rev
        displ_full_steps = displ_rev * full_steps_per_rev
        displ_microsteps = displ_full_steps / self.step_size 
        displ_microsteps = displ_microsteps.astype(int)
        return displ_microsteps

    def _integrate_steps(self, n_steps_actuated, signs):
        steps_moved = n_steps_actuated * signs * step_size
        self.steps_from_origin += steps_moved
        self.loc += steps_moved / self.full_steps_per_rev * self.cm_per_rev

    def _test_microcontroller_done(self, *args, **kwargs):
        # check if any data has returned from the microcontroller interface
        packet_rx = self.pos_uctrl_iface.data_available()

        # remember to actually read the data out of the buffer in an '_end' function
        return packet_rx

    def _test_new_target_set_remotely(self, *args, **kwargs):
        pass

    ##### State transition functions #####
    def _start_go_to_origin(self):
        self.pos_uctrl_iface.start_continuous_move(-10000, -10000, -10000)

    def _end_go_to_origin(self):
        steps_actuated = self.pos_uctrl_iface.end_continuous_move()

        self.loc = np.zeros(3)
        self.steps_from_origin = np.zeros(3)

    def _start_move_target(self):
        # calc number of steps from current pos to target pos
        displ_microsteps = self._calc_steps_to_pos(self._gen_int_target_pos)

        # send command to initiatem movement 
        self.pos_uctrl_iface.start_continuous_move(*displ_microsteps)

    def _end_move_target(self):
        # send command to kill motors
        steps_actuated = self.pos_uctrl_iface.end_continuous_move()

    ### Old functions ###
    def go_to_origin(self):
        '''
        Tap the origin limit switches so the absolute position of the target can be estimated. 
        Run at initialization and/or periodically to correct for any accumulating errors.
        '''
        steps_moved = np.array(self.pos_uctrl_iface.continuous_move(-10000, -10000, 10000))
        step_signs = np.array([-1, -1, 1], dtype=np.float64) * self.step_size

        if not np.any(np.isnan(self.steps_from_origin)): # error accumulation correction
            self.steps_from_origin += step_signs * steps_moved

            # if no position errors were accumulated, then self.steps_from_origin should all be 0
            acc_error = self.steps_from_origin
            print "accumulated step errors"
            print acc_error

        self.loc = np.zeros(3)
        self.steps_from_origin = np.zeros(3)

    def go_to_position(self, target_pos):
        if np.any(np.isnan(self.loc)):
            raise Exception("System must be 'zeroed' before it can go to an absolute location!")

        displ_cm = target_pos - self.loc

        # compute the number of steps needed to travel the desired displacement
        displ_rev = displ_cm / self.cm_per_rev
        displ_full_steps = displ_rev * full_steps_per_rev
        displ_microsteps = displ_full_steps / self.step_size

        steps_moved = np.array(self.pos_uctrl_iface.continuous_move(*displ_microsteps))
        steps_moved = steps_moved * np.sign(displ_microsteps) * self.step_size

        self.steps_from_origin += steps_moved
        self.loc += steps_moved / self.full_steps_per_rev * self.cm_per_rev
