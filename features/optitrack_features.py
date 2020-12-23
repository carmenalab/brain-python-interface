'''
Features for the Optitrack motiontracker
'''

from riglib.experiment import traits
from riglib.optitrack_client import optitrack
from datetime import datetime
import numpy as np
import os

TESTING_OFFSET = [0, 0.1, -0.22] # optitrack m [forward, up, right]
TESTING_SCALE = 100 # optitrack m --> screen cm

########################################################################################################
# Optitrack datasources
########################################################################################################
class Optitrack(traits.HasTraits):
    '''
    Enable reading of raw motiontracker data from Optitrack system
    Requires the natnet library from https://github.com/leoscholl/python_natnet
    To be used as a feature with the ManualControl task for the time being. However,
    ideally this would be implemented as a decoder :)
    '''

    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    smooth_features = traits.Int(1, desc="How many features to average")
    scale = traits.Float(TESTING_SCALE, desc="Control scale factor")
    offset = traits.Array(value=TESTING_OFFSET, desc="Control offset")

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the natnet client and recording
        import natnet
        client = natnet.Client.connect(logger=Logger())
        now = datetime.now()
        session = "C:/Users/Orsborn Lab/Documents/OptiTrack/Session " + now.strftime("%Y-%m-%d")
        take = now.strftime("Take %Y-%m-%d %H:%M:%S")
        client = natnet.Client.connect(logger=Logger())
        client.set_session(session)
        client.set_take(take)
        self.filename = os.path.join(session, take)
        client.start_recording() # TODO set LiveMode
        self.client = client

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        super().init()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method starts the motiondata source and stops it after the FSM has finished running
        '''
        self.motiondata.start()
        try:
            super().run()
        finally:
            self.motiondata.stop()

    def join(self):
        '''
        See riglib.experiment.Experiment.join(). Re-join the motiondata source process before cleaning up the experiment thread
        '''
        self.motiondata.join()
        super().join()

    def _start_None(self):
        '''
        Code to run before the 'None' state starts (i.e., the task stops)
        '''
        self.motiondata.stop()
        super()._start_None()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the optitrack recorded file into the database
        '''
        super().cleanup(database, saveid, **kwargs)
        if saveid is not None:
            self.client.stop_recording()
            database.save_data(self.filename, "optitrack", saveid, False, False) # Make sure you actually have an "optitrack" system added!
            print("Saved optitrack file to database")

    def _get_manual_position(self):
        ''' Overridden method to get input coordinates based on motion data'''

        # Get data from optitrack datasource
        data = self.motiondata.get() # List of (list of features)
        if len(data) == 0: # Data is not being streamed
            return
        recent = data[-self.smooth_features:] # How many recent coordinates to average
        averaged = np.mean(recent, axis=0) # List of averaged features
        coords = np.concatenate((averaged[0], [1])) # Take only the first feature
        if np.isnan(coords).any(): # No usable coords
            return
        return coords

class OptitrackSimulate(Optitrack):
    '''
    Fake optitrack data for testing
    '''
    
    @classmethod
    def pre_init(cls, saveid):
        super(Optitrack, cls).pre_init(saveid=saveid)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the fake natnet client
        self.client = optitrack.SimulatedClient()

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))

        # Save to the sink
        from riglib import sink
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.register(self.motiondata)
        super(Optitrack, self).init()

# Helper class for natnet logging
import logging
class Logger(object):

    def __init__(self):
        logging.basicConfig(filename='../log/optitrack.log')

    debug = logging.debug
    info = logging.info
    warning = logging.warning
    error = logging.error
    fatal = logging.critical
