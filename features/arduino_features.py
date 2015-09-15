'''
'''
import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os
import subprocess
from riglib import bmi
from riglib.bmi import extractor
from riglib.experiment import traits
from hdf_features import SaveHDF
import sys
import glob
import datetime
import serial

import config
import time

sec_per_min = 60

class PlexonSerialDIORowByte(object):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    def __init__(self, *args, **kwargs):
        super(PlexonSerialDIORowByte, self).__init__(*args, **kwargs)
        self.file_ext = kwargs.pop('file_ext', '.plx')
        self.data_root = kwargs.pop('data_root', '/storage/plexon/')
        self.file_pattern = os.path.join(self.data_root, '*' + self.file_ext)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the NIDAQ card as a sink
        '''
        from riglib import sink
        self.nidaq = sink.sinks.start(self.ni_out)
        super(PlexonSerialDIORowByte, self).init()

        # Find all the plexon files modified in the last day
        file_pattern = self.file_pattern
        file_names = glob.glob(file_pattern)
        start_time = datetime.datetime.today() - datetime.timedelta(days=1)
        file_names = filter(lambda fname: datetime.datetime.fromtimestamp(os.stat(fname).st_mtime) > start_time, file_names)

        self.possible_filenames = file_names
        self.possible_filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])

    @property
    def ni_out(self):
        '''
        Specify the output interface; can be overridden in child classes as long as 
        this method returns a class which has the same instance methods (close, register, send, sendMsg, etc.)
        '''
        # TODO ni_out ---> iface
        from riglib import serial_dio
        return serial_dio.SendRowByte

    
    @property
    def plexfile(self):
        '''
        Calculates the plexon file that's most likely associated with the current task
        based on the time at which the task ended and the "last modified" time of the 
        plexon files located at /storage/plexon/
        '''
        if hasattr(self, '_plexfile'):
            return self._plexfile
        else:
            ## Ideally, the correct 'plexfile' will be the only file whose filesize has changed since the 'init' function ran..
            filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
            inds, = np.nonzero(filesizes - self.possible_filesizes)
            if len(inds) == 1:
                print "only one plx file changed since the start of the task."
                self._plexfile = self.possible_filenames[inds[0]]
                return self._plexfile

            ## Otherwise, try to find a file whose last modified time is within 60 seconds of the task ending; this requires fairly accurate synchronization between the two machines
            if len(self.event_log) < 1:
                self._plexfile = None
                return self._plexfile
            
            start = self.event_log[-1][2]
            files = "/storage/plexon/*.plx"
            files = sorted(glob.glob(files), key=lambda f: abs(os.stat(f).st_mtime - start))
            
            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < sec_per_min:
                    self._plexfile = files[0]
                    return self._plexfile

            ## If both methods fail, return None; cleanup should warn the user that they'll have to link the plexon file manually
            self._plexfile = None
            return self._plexfile

    
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method stops the NIDAQ sink after the FSM has stopped running.
        '''
        try:
            super(PlexonSerialDIORowByte, self).run()
        finally:
            # Stop the NIDAQ sink
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        '''
        Extension of riglib.experiment.Experiment.set_state. Send the name of the next state to 
        plexon system and then proceed to the upstream set_state tasks.

        Parameters
        ----------
        condition : string
            Name of new state.
        **kwargs : dict 
            Passed to 'super' set_state function

        Returns
        -------
        None
        '''
        self.nidaq.sendMsg(condition)
        super(PlexonSerialDIORowByte, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''
        super(PlexonSerialDIORowByte, self).cleanup(database, saveid, **kwargs)

        # Sleep time so that the plx file has time to save cleanly        
        time.sleep(2)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if self.plexfile is not None:
            if dbname == 'default':
                database.save_data(self.plexfile, "plexon", saveid, True, False)
            else:
                database.save_data(self.plexfile, "plexon", saveid, True, False, dbname=dbname)
        else:
            print '\n\nPlexon file not found properly! It will have to be manually linked!\n\n'
        
    @classmethod 
    def pre_init(cls, saveid=None):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            port = serial.Serial('/dev/arduino_neurosync', baudrate=9600)
            port.write('p')
            time.sleep(0.5)
            port.write('r')
            port.close()

            time.sleep(3)
            super(PlexonSerialDIORowByte, cls).pre_init(saveid=saveid)

class TDTSerialDIORowByte(PlexonSerialDIORowByte):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    def __init__(self, *args, **kwargs):
        super(TDTSerialDIORowByte, self).__init__(*args, **kwargs)
        self.file_ext = kwargs.pop('file_ext', '.tev')
        self.data_root = kwargs.pop('data_root', '/storage/tdt/')
        self.file_pattern = os.path.join(self.data_root, '*' + self.file_ext)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the NIDAQ card as a sink
        '''
        from riglib import sink
        self.nidaq = sink.sinks.start(self.ni_out)
        super(TDTSerialDIORowByte, self).init()
        '''
        # Find all the plexon files modified in the last day
        file_pattern = self.file_pattern
        file_names = glob.glob(file_pattern)
        start_time = datetime.datetime.today() - datetime.timedelta(days=1)
        file_names = filter(lambda fname: datetime.datetime.fromtimestamp(os.stat(fname).st_mtime) > start_time, file_names)

        self.possible_filenames = file_names
        self.possible_filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
        '''

    @property
    def ni_out(self):
        '''
        Specify the output interface; can be overridden in child classes as long as 
        this method returns a class which has the same instance methods (close, register, send, sendMsg, etc.)
        '''
        # TODO ni_out ---> iface
        from riglib import serial_dio
        return serial_dio.SendRowByte

    
    @property
    def plexfile(self):
        '''
        Calculates the plexon file that's most likely associated with the current task
        based on the time at which the task ended and the "last modified" time of the 
        plexon files located at /storage/plexon/
        '''
        """
        if hasattr(self, '_plexfile'):
            return self._plexfile
        else:
            ## Ideally, the correct 'plexfile' will be the only file whose filesize has changed since the 'init' function ran..
            filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
            inds, = np.nonzero(filesizes - self.possible_filesizes)
            if len(inds) == 1:
                print "only one tdt file changed since the start of the task."
                self._plexfile = self.possible_filenames[inds[0]]
                return self._plexfile

            ## Otherwise, try to find a file whose last modified time is within 60 seconds of the task ending; this requires fairly accurate synchronization between the two machines
            if len(self.event_log) < 1:
                self._plexfile = None
                return self._plexfile
            
            start = self.event_log[-1][2]
            files = "/storage/tdt/*.tev"
            files = sorted(glob.glob(files), key=lambda f: abs(os.stat(f).st_mtime - start))
            
            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < sec_per_min:
                    self._plexfile = files[0]
                    return self._plexfile

            ## If both methods fail, return None; cleanup should warn the user that they'll have to link the plexon file manually
            self._plexfile = None
            return self._plexfile
        """
        pass

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method stops the NIDAQ sink after the FSM has stopped running.
        '''
        try:
            super(TDTSerialDIORowByte, self).run()
        finally:
            # Stop the NIDAQ sink
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        '''
        Extension of riglib.experiment.Experiment.set_state. Send the name of the next state to 
        plexon system and then proceed to the upstream set_state tasks.

        Parameters
        ----------
        condition : string
            Name of new state.
        **kwargs : dict 
            Passed to 'super' set_state function

        Returns
        -------
        None
        '''
        self.nidaq.sendMsg(condition)
        super(TDTSerialDIORowByte, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''
        # Stop recording
        # import comedi
        import config
        import time

        # com = comedi.comedi_open("/dev/comedi0")
        # comedi.comedi_dio_bitfield2(com, 0, 16, 16, 16)

        # port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=9600)
        # port.write('p')
        # port.close()

        super(TDTSerialDIORowByte, self).cleanup(database, saveid, **kwargs)

        # Sleep time so that the plx file has time to save cleanly
        
        time.sleep(2)
        """
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if self.plexfile is not None:
            if dbname == 'default':
                database.save_data(self.plexfile, "plexon", saveid, True, False)
            else:
                database.save_data(self.plexfile, "plexon", saveid, True, False, dbname=dbname)
        else:
            print '\n\nTDT file not found properly! It will have to be manually linked!\n\n'
        """
        
    @classmethod 
    def pre_init(cls, saveid=None):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=115200)
            # for k in range(5):
            port.write('p')
            time.sleep(0.5)
            port.write('r')
            port.close()

            time.sleep(3)
            super(TDTSerialDIORowByte, cls).pre_init(saveid=saveid)