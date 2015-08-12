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
    z={0:1, -1:1, 1:0},
    ) # convention flipped for z-stage

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
        print self.port.readline()
        print self.port.readline()
        self.sleep_motors()

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


