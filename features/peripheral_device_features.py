'''
Peripheral interface device features
'''

import time
import tempfile
import random
import traceback
import numpy as np
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
    def register_num_channels(self):
        from riglib import arduino_joystick, source, sink
        System = arduino_joystick.make(2, 1)
        self.joystick = source.DataSource(System)

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
