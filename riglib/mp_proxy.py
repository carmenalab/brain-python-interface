import os
import sys
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm
import ctypes

import numpy as np


class FuncProxy(object):
    '''
    Interface for calling functions in remote processes.
    '''
    def __init__(self, name, pipe, event=None):
        '''
        Constructor for FuncProxy

        Parameters
        ----------
        name : string
            Name of remote function to call
        pipe : mp.Pipe instance
            multiprocessing pipe through which to send data (function name, arguments) and receive the result
        event : mp.Event instance
            A flag to set which is multiprocessing-compatible (visible to both the current and the remote processes)

        Returns
        -------
        FuncProxy instance
        '''
        self.pipe = pipe
        self.name = name
        self.event = event

    def __call__(self, *args, **kwargs):
        '''
        Return the result of the remote function call

        Parameters
        ----------
        *args, **kwargs : positional arguments, keyword arguments
            To be passed to the remote function associated when the object was created

        Returns
        -------
        function result
        '''
        self.pipe.send((self.name, args, kwargs))
        if not self.event is None:
            self.event.set()
        return self.pipe.recv()