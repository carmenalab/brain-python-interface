from riglib.experiment import traits, Sequence
from riglib import tabletstream
from .target_graphics import *
from .target_capture_task import plantlist

import numpy as np

RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)

class DexterityMaze(WindowDispl2D, Sequence, traits.HasTraits):
    
    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", stop=None),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )

    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']

    #initial state
    state = "wait"
    reward_time = 1.
    target_color = (1,0,0,.5)
    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)

    limit2d = 1

    sequence_generators = ['test_maze']
    is_bmi_seed = True
    _target_color = RED

    # Runtime settable traits
    hold_time = traits.Float(0., desc="Length of hold required at targets")
    timeout_time = traits.Float(15, desc="Time allowed to go between targets")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')

    # Use this plant type 
    plant_type = 'cursor_14x14'
    cursor_radius = 0.1
    target_radius = 0.15; 
    _target_color = (1., 0., 0., 0.5)
    timeout_penalty_time = 1. 
    hold_penalty_time = 1.

    def __init__(self, *args, **kwargs):
        super(DexterityMaze, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        # Initialize the plant
        self.plant_vis_prev = True

        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)
                pass

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target3 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target4 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)

            self.targets = [target1, target2, target3, target4]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
                    pass
        
        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

        self.tablet_cursor = np.array([0.,0.])
        self.tablet = tabletstream.TabletSystem()
        self.tablet.start()

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        self.add_dtype('tablet_cursor', 'f8', (2, ))
        super(DexterityMaze, self).init()
        

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        self.move_effector()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        ### pull tablet data
        self.tablet_cursor = self.tablet.get()
        #print(self.tablet_cursor)

        try:
            self.task_data['tablet_cursor'] = self.tablet_cursor.copy()
        except:
            self.task_data['tablet_cursor'] = np.array([np.nan, np.nan])

        super(DexterityMaze, self)._cycle()

    def move_effector(self):
        try:
            self.plant.set_intrinsic_coordinates(np.array([self.tablet_cursor[0], 0., self.tablet_cursor[1]]))
        except:
            print('skipping updates')
            pass

    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        try:
            super(DexterityMaze, self).run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=True)
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
                self.show_object(self.cursor, show=False)

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(DexterityMaze, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)

    #### TEST FUNCTIONS ####
    def _test_start_trial(self, ts):
        if ts >= 1.:
            return True
        else:
            return False

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        #print('Entered target %.4f' %d)

        return d <= self.target_radius
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius
        return d > rad

    def _test_hold_complete(self, ts):
        return ts>=self.hold_time

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts>self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def _test_reward_end(self, ts):
        return ts>self.reward_time

    #### STATE FUNCTIONS ####
    def _parse_next_trial(self):
        self.targs, self.maze = self.next_trial

    def _start_wait(self):
        super(DexterityMaze, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()

        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index]
        self.target_location = self.targs[self.target_index]
        self.maze_location = self.maze[self.target_index]
        target.move_to_position(self.target_location)
        target.cue_trial_start()

        # print('start target %d'%self.target_index)
        # print('update maze')
        self.update_maze(self.maze_location)
        
    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)

        if idx < self.chain_length: 
            target = self.targets[idx]
            target.move_to_position(self.targs[idx])
    
    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index].cue_trial_end_success()

    def _start_hold_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1
    
    def _start_timeout_penalty(self):
        #hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        #super(ManualControlMulti, self)._start_reward()
        self.update_maze(self.maze[0, :, :])
        self.targets[self.target_index].show()

    @staticmethod
    def test_maze(length=1000):
        
        maze = np.zeros((2, 3, 4))
        
        ### initial maze, all ones 
        maze[0, 2, :] = 1

        ### Motors to retract to clear the maze path: 
        maze[1, 2, 3] = 1
        maze[1, 2, 2] = 1
        maze[1, 2, 1] = 1
        maze[1, 1, 1] = 1
        maze = maze.astype(int)
        
        pairs = np.zeros([2,3])
        pairs[0, 0] = 2.23; 
        pairs[0, 2] = 0.58;
        
        pairs[1, 0] = 3.38# ? 
        pairs[1, 2] = 1.36 # ? 

        gen = []
        for i in range(length):
            gen.append([pairs, maze])
        return gen

    @staticmethod
    def test_maze2(length=1000):
        
        maze = np.zeros((2, 3, 4))
        maze[0, 2, :] = 1

        ### Motors to retract to clear the maze path: 
        maze[1, 2, 0] = 1
        maze[1, 1, 0] = 1
        maze[1, 1, 2] = 1
        maze[1, 2, 2] = 1
        maze[1, 2, 3] = 1
        maze[1, 0, 0] = 1
        maze = np.fliplr(maze)
        maze = maze.astype(int)
        
        pairs = np.zeros([2,3])
        pairs[0, 0] = 2.23; 
        pairs[0, 2] = 0.58;
        
        pairs[1, 0] = 2.81# ? 
        pairs[1, 2] = 0.58 # ? 

        gen = []
        for i in range(length):
            gen.append([pairs, maze])
        return gen

    @staticmethod
    def test_maze_combo(length=1000):
        maze = np.zeros((4, 3, 4))
        
        ### initial maze, all ones on the bottom
        maze[0, 2, :] = 1
        maze[2, 2, :] = 1

        ### Motors to retract to clear the maze path: 
        maze[1, 2, 3] = 1
        maze[1, 2, 2] = 1
        maze[1, 2, 1] = 1
        maze[1, 1, 1] = 1

        #### maze 2 -- 4th col all open 
        maze[3, 2, 3] = 1
        maze[3, 1, 3] = 1
        maze[3, 0, 3] = 1
        ### 2nd col all open 
        maze[3, 2, 1] = 1
        maze[3, 1, 1] = 1

        #### 1st col bottom oopen 
        maze[3, 2, 0] = 1
        maze = maze.astype(int)
        
        pairs = np.zeros([4,3])
        ### Starting position 
        pairs[0, 0] = 2.23; 
        pairs[0, 2] = 0.58;
        pairs[2, 0] = 2.23; 
        pairs[2, 2] = 0.58;
        
        ### End for maze `1
        pairs[1, 0] = 3.38# ? 
        pairs[1, 2] = 1.36 # ? 
        ### End for maze 2 
        pairs[3, 0] = 3.# ? 
        pairs[3, 2] = 0.58 # ? 

        gen = []
        for i in range(length):
            gen.append([pairs, maze])
        return gen
