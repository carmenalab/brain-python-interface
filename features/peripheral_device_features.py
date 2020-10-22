'''
Peripheral interface device features
'''

import time
import tempfile
import random
import traceback
import numpy as np
import pygame
import copy
import fnmatch
import os
import subprocess
from riglib.experiment import traits

###### CONSTANTS
sec_per_min = 60


class Joystick(object):
    '''
    Code to use an analog joystick with signals digitized by the phidgets board
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' instantiates a DataSource with 2 channels for the two analog 
        inputs from the phidgets joystick. 
        '''
        from riglib import source, phidgets, sink
        self.sinks = sink.sinks
        #System = phidgets.make(2, 1)
        #self.joystick = source.DataSource(System)

        self.register_num_channels()
        super(Joystick, self).init()
        self.sinks.register(self.joystick)

    def register_num_channels(self):
        from riglib import source, phidgets, sink
        System = phidgets.make(2, 1)
        self.joystick = source.DataSource(System)

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the joystick source and stops it after the FSM has finished running
        '''
        self.joystick.start()
        try:
            super(Joystick, self).run()
        finally:
            self.joystick.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.joystick.join()
        super(Joystick, self).join()

class ArduinoJoystick(Joystick):
    def init(self):
        '''
        Same as above, w/o Phidgets import
        '''
        from riglib import source, sink
        self.sinks = sink.sinks
        self.register_num_channels()
        super(Joystick, self).init()
        self.sinks.register(self.joystick)

    def register_num_channels(self):
        from riglib import arduino_joystick, source, sink
        System = arduino_joystick.make(2, 1)
        self.joystick = source.DataSource(System)

class ArduinoIMU(object):
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' instantiates a DataSource with 2 channels for the
        inputs from the IMU
        '''
        from riglib import sink
        self.sinks = sink.sinks
        self.register_num_channels()
        super(ArduinoIMU, self).init()
        self.sinks.register(self.arduino_imu)

    def register_num_channels(self):
        from riglib import source, arduino_imu
        System = arduino_imu.make(6, 1)
        self.arduino_imu = source.DataSource(System)

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the joystick source and stops it after the FSM has finished running
        '''
        self.arduino_imu.start()
        try:
            super(ArduinoIMU, self).run()
        finally:
            self.arduino_imu.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.arduino_imu.join()
        super(ArduinoIMU, self).join()    

class Joystick_plus_TouchSensor(Joystick):
    '''
    code to use touch sensor (attached to joystick in exorig) plus joystick
    '''
    def register_num_channels(self):
        from riglib import source, phidgets, sink
        System = phidgets.make(3, 1)
        self.joystick = source.DataSource(System)

class DualJoystick(object):
    '''
    A two-joystick interface, similar to Joystick
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates a 4-channel DataSource, two channels for each joystick
        -------
        '''
        from riglib import source, phidgets
        System = phidgets.make(4, 1)
        self.dualjoystick = source.DataSource(System)
        super(DualJoystick, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the dual_joystick source and stops it after the FSM has finished running
        '''
        self.dualjoystick.start()
        try:
            super(DualJoystick, self).run()
        finally:
            self.dualjoystick.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the joystick source process before cleaning up the experiment thread
        '''
        self.dualjoystick.join()
        super(DualJoystick, self).join()

class Button(object):
    '''
    Deprecated!

    Adds the ability to respond to the button, as well as to keyboard responses
    The "button" was a switch connected to a modified mouse so that the digital input went through 
    the mouse interface (hence the calls to pygame's mouse interface)
    '''
    def screen_init(self):
        super(Button, self).screen_init()
        import pygame
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def _get_event(self):
        import pygame
        btnmap = {1:1, 3:4}
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if btn.button in btnmap:
                return btnmap[btn.button]

        return super(Button, self)._get_event()
    
    def _while_reward(self):
        super(Button, self)._while_reward()
        import pygame
        pygame.event.clear()
    
    def _while_penalty(self):
        #Clear out the button buffers
        super(Button, self)._while_penalty()
        import pygame
        pygame.event.clear()
    
    def _while_wait(self):
        super(Button, self)._while_wait()
        import pygame
        pygame.event.clear()


class KeyboardControl(object):
    '''
    this class implements a python cursor control task for human
    '''

    def __init__(self, *args, **kwargs):
        self.move_step = 1
        self.assist_level = (0.5, 0.5)
        super(KeyboardControl, self).__init__(*args, **kwargs)
    
    # override the _cycle function
    def _cycle(self):
        self.move_effector_cursor()
        super(KeyboardControl, self)._cycle()

    def move_effector(self):
        pass

    def move_plant(self, **kwargs):
        pass

    # use keyboard to control the task
    def move_effector_cursor(self):
        curr_pos = copy.deepcopy(self.plant.get_endpoint_pos())

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.type == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    curr_pos[0] -= self.move_step
                if event.key == pygame.K_RIGHT:
                    curr_pos[0] += self.move_step
                if event.key == pygame.K_UP:
                    curr_pos[2] += self.move_step
                if event.key == pygame.K_DOWN:
                    curr_pos[2] -= self.move_step
            #print('Current position: ')
            #print(curr_pos)

        # set the current position
        self.plant.set_endpoint_pos(curr_pos)

    def _start_wait(self):
        self.wait_time = 0.
        super(KeyboardControl, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class MouseControl(KeyboardControl):

    def init(self, *args, **kwargs):
        self.pos = self.plant.get_endpoint_pos()
        super(MouseControl, self).init(*args, **kwargs)
    
    def move_effector_cursor(self):

        # Update position but keep mouse in center
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        rel = pygame.mouse.get_rel()
        self.pos[0] += rel[0] / self.window_size[0] * self.fov
        self.pos[2] -= rel[1] / self.window_size[1] * self.fov
        pos, _ = self.plant._bound(self.pos, [])
        self.plant.set_endpoint_pos(pos)

    def cleanup(self, *args, **kwargs):
        pygame.mouse.set_visible(True)
        super(MouseControl, self).cleanup(*args, **kwargs)

from .neural_sys_features import CorticalBMI
class MouseBMI(CorticalBMI):
    @property 
    def sys_module(self):
        from riglib import debug
        return debug