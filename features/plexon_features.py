'''
Features for interacting with Plexon's Omniplex neural recording system
'''
import numpy as np
import os
from .hdf_features import SaveHDF
import glob
import datetime

sec_per_min = 60

class RelayPlexon(object):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the NIDAQ card as a sink
        '''
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        self.nidaq = sink_manager.start(self.ni_out)
        super(RelayPlexon, self).init()

        # Find all the plexon files modified in the last day
        file_pattern = "/storage/plexon/*.plx"
        file_names = glob.glob(file_pattern)
        start_time = datetime.datetime.today() - datetime.timedelta(days=1)
        file_names = [fname for fname in file_names if datetime.datetime.fromtimestamp(os.stat(fname).st_mtime) > start_time]

        self.possible_filenames = file_names
        self.possible_filesizes = np.array([os.stat(fname).st_size for fname in self.possible_filenames])

    @property
    def ni_out(self):
        '''
        Specify the output interface; can be overridden in child classes as long as
        this method returns a class which has the same instance methods (close, register, send, sendMsg, etc.)
        '''
        from riglib.dio import nidaq
        return nidaq.SendAll

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
                print("only one plx file changed since the start of the task.")
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
            super(RelayPlexon, self).run()
        finally:
            # Stop the NIDAQ sink
            self.nidaq.stop()

            # Remotely stop the recording on the plexon box
            import comedi
            import time
            com = comedi.comedi_open("/dev/comedi0")
            time.sleep(0.5)
            comedi.comedi_dio_bitfield2(com, 0, 16, 16, 16)

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
        super(RelayPlexon, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''
        # Stop recording
        import comedi
        import time

        com = comedi.comedi_open("/dev/comedi0")
        comedi.comedi_dio_bitfield2(com, 0, 16, 16, 16)

        super(RelayPlexon, self).cleanup(database, saveid, **kwargs)

        # Sleep time so that the plx file has time to save cleanly
        time.sleep(2)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if self.plexfile is not None:
            if dbname == 'default':
                database.save_data(self.plexfile, "plexon", saveid, True, False)
            else:
                database.save_data(self.plexfile, "plexon", saveid, True, False, dbname=dbname)
        else:
            print('\n\nPlexon file not found properly! It will have to be manually linked!\n\n')

    @classmethod
    def pre_init(cls, saveid=None, **kwargs):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            import comedi
            import time

            com = comedi.comedi_open("/dev/comedi0")
            # stop any recording
            comedi.comedi_dio_bitfield2(com, 0, 16, 16, 16)
            time.sleep(0.1)
            # start recording
            comedi.comedi_dio_bitfield2(com, 0, 16, 0, 16)

            time.sleep(3)
            super(RelayPlexon, cls).pre_init(saveid=saveid, **kwargs)


class RelayPlexByte(RelayPlexon):
    '''
    Relays a single byte (0-255) to synchronize the rows of the HDF table(s) with the plexon recording clock.
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' ensures that this feature is
        only used if the SaveHDF feature is also enabled.
        '''
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayPlexByte feature only available with SaveHDF")
        super(RelayPlexByte, self).init()

    @property
    def ni_out(self):
        '''
        see documentation for RelayPlexon.ni_out
        '''
        from riglib.dio import nidaq
        return nidaq.SendRowByte


from .neural_sys_features import CorticalBMI
class PlexonBMI(CorticalBMI):
    @property
    def sys_module(self):
        from riglib import plexon
        return plexon

