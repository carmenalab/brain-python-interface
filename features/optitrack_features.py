'''
Features for the Optitrack motiontracker
'''

from riglib.experiment import traits
from riglib.optitrack_client import optitrack
from built_in_tasks.manualcontrolmultitasks import transformations
from datetime import datetime
import numpy as np
import os

TESTING_OFFSET = [0, -40, 0] # optitrack cm
TESTING_SCALE = 2 # optitrack cm --> screen cm

transformations = dict(
    testing = np.linalg.multi_dot((
        np.array(                       # Offset
            [[1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [TESTING_OFFSET[0], TESTING_OFFSET[1], TESTING_OFFSET[2], 1]]
        ),
        np.array(                       # Scale
            [[TESTING_SCALE, 0, 0, 0], 
            [0, TESTING_SCALE, 0, 0], 
            [0, 0, TESTING_SCALE, 0], 
            [0, 0, 0, 1]]
        ),
        np.array(                       # Rotation
            [[0, 0, 1, 0], 
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 1]]
        ),
    )),
)

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

    optitrack_recording = traits.Bool(True, desc="Automatically start/stop optitrack recording")
    optitrack_feature = traits.OptionsList(("rigid body", "skeleton", "marker"))
    optitrack_num_features = traits.Int(1, desc="How many features to average")
    transformation = traits.OptionsList(tuple(transformations.keys()), desc="Control transformation matrix")

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' sets up the DataSource for interacting with the 
        motion tracker system and registers the source with the SinkRegister so that the data gets saved to file as it is collected.
        '''

        # Start the natnet client
        import natnet
        self.client = natnet.Client.connect()

        # Create a source to buffer the motion tracking data
        from riglib import source
        self.motiondata = source.DataSource(optitrack.make(optitrack.System, self.client, self.optitrack_feature, self.optitrack_num_features))

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
        now = datetime.now()
        session = now.strftime("Session %Y-%m-%d")
        take = now.strftime("Take %Y-%m-%d %H:%M:%S")
        self.client.set_session(session)
        self.client.set_take(take)
        self.filename = os.path.join(session, take)
        if self.optitrack_recording:
            self.client.start_recording()

        self.motiondata.start()
        try:
            super().run()
        finally:
            self.client.stop_recording()
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
        if self.optitrack_recording:
            database.save_data(self.filename, "optitrack", saveid, False, False) # Make sure you actually have an "optitrack" system added!
            print("Saved optitrack file to database")

    def move_effector(self):
        ''' Overridden method to move the cursor based on motion data'''

        coords = self.motiondata.get()
        if len(coords) == 0:
            return
        coords = coords[-1] # Only use the last recorded coordinate
        if self.optitrack_num_features > 1:
            coords = np.mean(coords)
        coords = np.concatenate((np.squeeze(coords), [1]))
        coords = np.matmul(coords, transformations[self.transformation])

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