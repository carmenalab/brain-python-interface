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

    @classmethod
    def pre_init(cls, saveid):
        # Temporary code to start recording over the nets
        if saveid is not None:
            now = datetime.now()
            session = "C:/Users/Orsborn Lab/Documents/OptiTrack/Session " + now.strftime("%Y-%m-%d")
            take = now.strftime("Take %Y-%m-%d %H:%M:%S")
            
            import natnet
            client = natnet.Client.connect(logger=Logger())
            client.set_session(session)
            client.set_take(take)
            # TODO set LiveMode
            client.start_recording()

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the natnet client
        import natnet
        self.client = natnet.Client.connect(logger=Logger())

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, 1))
        self.no_data_count = 0
        self.missed_frames = 0

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

    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the optitrack recorded file into the database
        '''
        super().cleanup(database, saveid, **kwargs)
        if saveid is not None:
            self.client.stop_recording()
            # Don't yet have this capability:
            # session = self.client.get_session()
            # filename = self.client.get_take()
            # database.save_data(filename, "optitrack", saveid, False, False) # Make sure you actually have an "optitrack" system added!
            # print("Saved optitrack file to database")

    def move_effector(self):
        ''' Overridden method to move the cursor based on motion data'''

        # Get data from optitrack datasource
        data = self.motiondata.get() # List of (list of features)
        if len(data) == 0: # Data is not being streamed
            self.no_data_count += 1
            self.missed_frames += 1
            self.reportstats['Frames w/o mocap'] = self.missed_frames 
            return
        recent = data[-self.smooth_features:] # How many recent coordinates to average
        averaged = np.mean(recent, axis=0) # List of averaged features
        coords = np.concatenate((averaged[0], [1])) # Take only the first feature

        if np.isnan(coords).any(): # No usable coords
            self.no_data_count += 1
            self.missed_frames += 1
            self.reportstats['Frames w/o mocap'] = self.missed_frames 
            return

        self.no_data_count = 0
        coords = self._transform_coords(coords)

        # Set y coordinate to 0 for 2D tasks
        if self.limit2d:
            coords[1] = 0

        # Set cursor position
        if not self.velocity_control:
            self.current_pt = coords[0:3]
        else:
            epsilon = 2*(10**-2) # Define epsilon to stabilize cursor movement
            if sum((coords[0:3])**2) > epsilon:

                # Add the velocity (units/s) to the position (units)
                self.current_pt = coords[0:3] / self.fps + self.last_pt
            else:
                self.current_pt = self.last_pt

        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.current_pt.copy()

class OptitrackSimulate(Optitrack):
    '''
    Fake optitrack data for testing
    '''

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
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, self.optitrack_num_features))

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
