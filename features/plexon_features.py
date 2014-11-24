'''
Features for interacting with Plexon's Omniplex neural recording system
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
        self.nidaq = sink.sinks.start(self.ni_out)
        super(RelayPlexon, self).init()

    @property
    def ni_out(self):
        ''' Docstring '''
        from riglib import nidaq
        return nidaq.SendAll

    @property
    def plexfile(self):
        '''
        Calculates the plexon file that's most likely associated with the current task
        based on the time at which the task ended and the "last modified" time of the 
        plexon files located at /storage/plexon/
        '''
        import os, sys, glob, time
        if len(self.event_log) < 1:
            return None
        
        start = self.event_log[-1][2]
        files = "/storage/plexon/*.plx"
        files = sorted(glob.glob(files), key=lambda f: abs(os.stat(f).st_mtime - start))
        
        if len(files) > 0:
            tdiff = os.stat(files[0]).st_mtime - start
            if abs(tdiff) < sec_per_min:
                 return files[0]
    
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method stops the NIDAQ sink after the FSM has stopped running.
        '''
        try:
            super(RelayPlexon, self).run()
        finally:
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.nidaq.sendMsg(condition)
        super(RelayPlexon, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(RelayPlexon, self).cleanup(database, saveid, **kwargs)
        time.sleep(2)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if self.plexfile is not None:
            if dbname == 'default':
                database.save_data(self.plexfile, "plexon", saveid, True, False)
            else:
                database.save_data(self.plexfile, "plexon", saveid, True, False, dbname=dbname)
        
class RelayPlexByte(RelayPlexon):
    '''
    Relays a single byte (0-255) to synchronize the rows of the HDF table(s) with the plexon recording clock.
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' ensures that this feature is only used if the SaveHDF feature is also enabled.
        '''
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayPlexByte feature only available with SaveHDF")
        super(RelayPlexByte, self).init()

    @property
    def ni_out(self):
        '''
        see documentation for RelayPlexon.ni_out 
        '''
        from riglib import nidaq
        return nidaq.SendRowByte


class PlexonData(traits.HasTraits):
    '''Stream Plexon neural data'''
    plexon_channels = None

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates an appropriate DataSource for either Spike, LFP, or auxiliary analog 
        data (depends on the type of feature extractor used by the decoder).
        '''
        from riglib import plexon, source

        if hasattr(self.decoder, 'extractor_cls'):
            if 'spike' in self.decoder.extractor_cls.feature_type:  # e.g., 'spike_counts'
                self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)
            elif 'lfp' in self.decoder.extractor_cls.feature_type:  # e.g., 'lfp_power'
                self.neurondata = source.MultiChanDataSource(plexon.LFP, channels=self.plexon_channels)
            elif 'emg' in self.decoder.extractor_cls.feature_type:  # e.g., 'emg_amplitude'
                self.neurondata = source.MultiChanDataSource(plexon.Aux, channels=self.plexon_channels)
            else:
                raise Exception("Unknown extractor class, unable to create data source object!")
        else:
            # if using an older decoder that doesn't have extractor_cls (and 
            # extractor_kwargs) as attributes, then just create a DataSource 
            # with plexon.Spikes by default 
            # (or if there is no decoder at all in this task....)
            self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)

        super(PlexonData, self).init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the 'neurondata' source before the FSM begins execution
        and stops it after the FSM has completed. 
        '''
        self.neurondata.start()
        try:
            super(PlexonData, self).run()
        finally:
            self.neurondata.stop()

class PlexonBMI(PlexonData):
    '''
    Special case of PlexonData which specifies a subset of channels to stream, i.e., the ones used by the Decoder
    May not be available for all recording systems. 
    '''
    decoder = traits.Instance(bmi.Decoder)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets the channels to be the channels of the decoder
        so that the PlexonData source only grabs the channels actually used by the decoder. 
        '''
        self.plexon_channels = self.decoder.units[:,0]
        super(PlexonBMI, self).init()
