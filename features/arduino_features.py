'''
Features to include when using the Arduino board to remotely start neural
recording for Plexon/Blackrock/TDT systems and also synchronize data between the task and the neural recordings
'''
import time
import numpy as np
import os
from riglib import serial_dio
import glob
import datetime
import serial

sec_per_min = 60
baudrate = 115200 #9600

class SerialDIORowByte(object):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for SerialDIORowByte

        Parameters
        ----------
        None

        Returns
        -------
        SerialDIORowByte instance
        '''
        super(SerialDIORowByte, self).__init__(*args, **kwargs)

        self.checked_for_file_changes = False
        self.task_start_time = time.time()
        self.possible_filesizes = []
        self.possible_filenames = []

    def filter_files(self, file_names, start_time=datetime.datetime.today() - datetime.timedelta(days=1)):
        '''
        Filter a list of filenames to find the ones which were timestamped after a particular start_time

        Parameters
        ----------
        file_names : iterable
            Each element should be a string of a filename which exists
        start_time : datetime.datetime object, optional, default = 1 day ago
            Cutoff time for the file timestamp. The file needs
            to be modified after this time in order to pass the filter.
        '''
        return [fname for fname in file_names if datetime.datetime.fromtimestamp(os.stat(fname).st_mtime) > start_time]

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the NIDAQ card as a sink
        '''
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        self.nidaq = sink_manager.start(serial_dio.SendRowByte)
        self.dbq_kwargs = dict()
        super(SerialDIORowByte, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running.
        See riglib.experiment.Experiment.run(). This 'run' method stops the NIDAQ sink after the FSM has stopped running.
        '''
        try:
            super(SerialDIORowByte, self).run()
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
        super(SerialDIORowByte, self).set_state(condition, **kwargs)

    def _cycle(self):
        if not self.checked_for_file_changes and (time.time() - self.task_start_time > 5):
            filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
            inds, = np.nonzero(filesizes - self.possible_filesizes)
            if len(inds) == 0:
                print("\n\n\nFile recording did not start? no files have changed size!\n")
            else:
                print("%d neural files have changed since the start of the block" % len(inds))
            self.checked_for_file_changes = True
        super(SerialDIORowByte, self)._cycle()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''

        # write the 'stop' command to the port just to be more sure that neural recording has finished.
        port = serial.Serial('/dev/arduino_neurosync', baudrate=baudrate)
        port.write("p")
        super(SerialDIORowByte, self).cleanup(database, saveid, **kwargs)

        # Sleep time so that the neural recording system has time to save cleanly
        time.sleep(5)

        print("Beginning neural data file cleanup")
        # specify which database to save to. If you're running from the web interface, this will always pick the 'default' database

        if "dbname" in kwargs:
            self.dbq_kwargs['dbname']=kwargs["dbname"]

        # Call the appropriate functions in the dbq module to actually link the files
        if self.data_files is None or len(self.data_files) == 0:
            print("\tData files not found properly!\n\tThey will have be manually linked using dbq.save_data!\n\n")
        elif isinstance(self.data_files, str):
            database.save_data(self.data_files, self.db_sys_name, saveid, True, False, **self.dbq_kwargs)
        elif np.iterable(self.data_files):
            for df in self.data_files:
                ext = os.path.splitext(df)[1][1:]
                database.save_data(df, self.db_sys_name, saveid, True, False, ext, **self.dbq_kwargs)

    @classmethod
    def pre_init(cls, saveid=None, **kwargs):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            port = serial.Serial('/dev/arduino_neurosync',baudrate=baudrate)
            port.write('p')
            time.sleep(0.5)
            port.write('r')
            time.sleep(3)

            port.close()
            super(SerialDIORowByte, cls).pre_init(saveid=saveid, **kwargs)

class PlexonSerialDIORowByte(SerialDIORowByte):
    db_sys_name = "plexon"
    storage_root = '/storage/plexon/'

    def init(self):
        """
        Find all the plx files created in the last 24 h before calling the parents init method
        """
        # Find all the plexon files modified in the last day
        file_pattern = self.storage_root + "*.plx"
        self.possible_filenames = self.filter_files(glob.glob(file_pattern))
        self.possible_filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])

        super(PlexonSerialDIORowByte, self).init()

    @property
    def data_files(self):
        '''
        Calculates the plexon file that's most likely associated with the current task
        based on the time at which the task ended and the "last modified" time of the
        plexon files located at /storage/plexon/
        '''
        if hasattr(self, '_data_files'):
            return self._data_files
        else:
            ## Ideally, the correct 'data_files' will be the only file whose filesize has changed since the 'init' function ran..
            filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
            inds, = np.nonzero(filesizes - self.possible_filesizes)
            if len(inds) == 1:
                print("\tonly one plx file changed since the start of the task.")
                self._data_files = self.possible_filenames[inds[0]]
                return self._data_files

            ## Otherwise, try to find a file whose last modified time is within 60 seconds of the task ending; this requires fairly accurate synchronization between the two machines
            if len(self.event_log) < 1:
                self._data_files = None
                return self._data_files

            start = self.event_log[-1][2]
            #files = "/storage/plexon/*.plx"
            files = "/storage/plexon/test*.plx" #Only use files that have not yet been renamed
            files = sorted(glob.glob(files), key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < sec_per_min:
                    print("\tfound plexon file by finding a file with a timestamp within one minute of the last thing in the event log")
                    self._data_files = files[0]
                    return self._data_files

            ## If both methods fail, return None; cleanup should warn the user that they'll have to link the plexon file manually
            self._data_files = None
            return self._data_files


class BlackrockSerialDIORowByte(SerialDIORowByte):
    db_sys_name = "blackrock2"
    storage_root = "/storage/blackrock"

    file_exts = ["*.nev", "*.ns1", "*.ns2", "*.ns3", "*.ns4", "*.ns5", "*.ns6"]

    def init(self):
        self.possible_filenames = []
        for file_ext in self.file_exts:
            file_pattern1 = os.path.join(self.storage_root, "*/", file_ext)
            file_pattern2 = os.path.join(self.storage_root, file_ext)

            self.possible_filenames += self.filter_files(glob.glob(file_pattern1))
            self.possible_filenames += self.filter_files(glob.glob(file_pattern2))

        self.possible_filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])

        super(BlackrockSerialDIORowByte, self).init()

    @property
    def data_files(self):
        if not hasattr(self, "_data_files"):
            ## Ideally, the correct 'data_files' will be the only file whose filesize has changed since the 'init' function ran..
            filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])
            inds, = np.nonzero(filesizes - self.possible_filesizes)

            files_which_changed_size = [self.possible_filenames[k] for k in inds]

            # make sure that all the base filenames are the same
            file_dates = [x.split(".")[0] for x in files_which_changed_size]
            if len(np.unique(file_dates)) <= 1:
                self._data_files = files_which_changed_size
            else:
                print("Filenames ambiguous! BlackrockSerialDIORowByte cannot figure out which files are associated with this block")
                print("these are the options:")
                print(files_which_changed_size)
                print()

        return self._data_files


class TDTSerialDIORowByte(SerialDIORowByte):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    db_sys_name = "tdt"
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
        sink_manager = sink.SinkManager.get_instance()
        self.nidaq = sink_manager.start(self.ni_out)
        super(TDTSerialDIORowByte, self).init()

    @property
    def ni_out(self):
        '''
        Specify the output interface; can be overridden in child classes as long as
        this method returns a class which has the same instance methods (close, register, send, sendMsg, etc.)
        '''
        # TODO ni_out ---> iface
        from riglib import serial_dio
        return serial_dio.SendRowByte

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
    def pre_init(cls, saveid=None, **kwargs):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            port = serial.Serial('/dev/arduino_neurosync', baudrate=baudrate)
            # for k in range(5):
            port.write('p')
            time.sleep(0.5)
            port.write('r')
            port.close()

            time.sleep(3)
            super(TDTSerialDIORowByte, cls).pre_init(saveid=saveid, **kwargs)

    @property
    def data_files(self):
        return None
