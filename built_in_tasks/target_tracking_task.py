'''
A generic target tracking task
'''
import numpy as np
import time
import os
import math
import traceback
import random
from collections import OrderedDict

from riglib.experiment import traits, Sequence, FSMTable, StateTransitions
from riglib.stereo_opengl import ik
from riglib import plants

from riglib.stereo_opengl.window import Window
from .target_graphics import *

## Plants
# List of possible "plants" that a subject could control either during manual or brain control
cursor = plants.CursorPlant()
shoulder_anchor = np.array([2., 0., -15])
chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))
chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)

plantlist = dict(
    cursor=cursor,
    chain_20_20=chain_20_20,
    chain_20_20_endpt=chain_20_20_endpt)

class TargetTracking(Sequence):
    '''
    This is a generic cued target tracking skeleton, to form as a common ancestor to the most
    common type of motor control tracking tasks.
    '''
    status = dict(
        wait = dict(start_trial="initiation"),
        initiation = dict(start_tracking="tracking"),
        tracking = dict(success="reward", timeout="timeout_penalty"),
        timeout_penalty = dict(timeout_penalty_end = "wait", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    # initial state
    state = "wait"
    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    penalty_time = traits.Float(.5, desc="Length of penalty")
    max_distance_error = traits.Float(2, desc="Maximum deviation from the trajectory for reward (cm)")
    
    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8',(3,)), ('disturbance_path', 'f8',(3,)), ('is_disturbance', '?')])
        super().init()

        self.frame_index = 0 # index in the frame in a trial
        self.total_distance_error = 0 # Euclidian distance between cursor and target during each trial
        self.trial_timed_out = True # check if the trial is finished
        self.plant_position = []
        self.disturbance_trial = False

    def _parse_next_trial(self):
        '''Get the required data from the generator'''
        self.gen_indices, self.targs, self.disturbance_trial, self.disturbance_path = self.next_trial
        
        self.targs = np.squeeze(self.targs,axis=0)
        self.disturbance_path = np.squeeze(self.disturbance_path)

        for i in range(len(self.disturbance_path)):
            # Update the data sinks with trial information
            self.trial_record['trial'] = self.calc_trial_num()
            self.trial_record['index'] = self.gen_indices
            self.trial_record['target'] = self.targs[i]
            self.trial_record['disturbance_path'] = self.disturbance_path[i]
            self.trial_record['is_disturbance'] = self.disturbance_trial
            self.sinks.send("trials", self.trial_record)

        # if self.disturbance_trial:
        #     print("Disturbance trial")

    def _start_wait(self):
        # Call parent method to draw the next target capture sequence from the generator
        super()._start_wait()

        #saved plant poitions
        self.plant_position = []

    def _start_initiation(self):
        self.frame_index = 0
        self.trajectory.move_to_position(np.array([-self.frame_index/3,0,0]))
        self.target.move_to_position(np.array([0,0,self.targs[self.frame_index+30][2]]))

    def _while_initiation(self):
        '''Nothing generic to do.'''
        pass

    def _end_initiation(self):
        '''Nothing generic to do.'''
        pass

    def _start_tracking(self):
        self.trial_timed_out = False
        self.total_distance_error = 0

    def _while_tracking(self):
        # Calculate and sum distance between center of cursor and current target position
        self.total_distance_error += self.test_in_tracking() 
        
        # Move Target and trajectory to next frame so it appears to be moving
        self.update_frame()
        
        # Check if the trial is over and there are no more target frames to display
        if self.frame_index + 30 >= np.shape(self.targs)[0]:
            self.trial_timed_out = True
    
    def update_frame(self):
        self.trajectory.move_to_position(np.array([-self.frame_index/3,0,0]))
        #if self.frame_index >= 0: #Offset tracker to move in sync with trajectory
        self.target.move_to_position(np.array([0,0,self.targs[self.frame_index+30][2]]))
        self.trajectory.show()
        self.target.show()
        self.frame_index +=1

    def _end_tracking(self):
        '''Nothing generic to do.'''
        pass

    def _while_reward(self):
        '''Nothing generic to do.'''
        pass

    def _end_reward(self):
        '''Nothing generic to do.'''
        pass
    
    def _start_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    ################## State transition test functions ##################
    def _test_start_trial(self, time_in_state):
        '''Start next trial automatically. You may want this to instead be
            - a random delay
            - require some initiation action
        '''
        return False

    def _test_start_tracking(self, time_in_state):
        '''
        Test if the cursor is inside the target to initiate trials. 
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target.get_position())
        return d <= (self.target_radius - self.cursor_radius)

    def _test_success(self, time_in_state):
        '''
        Test if the current state is successful and should transition to reward state.
        In order to be successful it needs to be the end of the trial and the total 
        distance error must be less than max_distance_error cm on average of the whole trial.
        This means the center of the cursor was on average within max_distance_error cm of the center of the target.
        '''
        return self.trial_timed_out and self.total_distance_error/self.frame_index < self.max_distance_error

    def _test_timeout(self, time_in_state):
        '''
        This test if the trial has finshed unsuccessfully and should transition to the wait state
        _test_success() is called before this test function and so if the trial is over, but successful
        it would transition to reward otherwise it will transition to the penalty state.
        '''
        return  self.trial_timed_out or self.pause

    def _test_reward_end(self, time_in_state):
        '''
        Test the reward state has ended
        '''
        return time_in_state > self.reward_time
    
    def _test_timeout_penalty_end(self, time_in_state):
        '''  
        Test the penalty state has ended.
        '''
        return time_in_state > self.penalty_time

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)
    
    def test_in_tracking(self):
        '''
        return the distance between center of cursor and cednter of the target
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target.get_position())
        if d <= (self.target_radius - self.cursor_radius):
            self.target.cue_trial_end_success()
        else:
            self.target.reset()
        return d
    
    def add_disturbance(self, current_position, current_velocity, disturbance, prev_disturbance):
        if self.limit1d:
            return current_position + current_velocity + [0,0,(disturbance - prev_disturbance)]
        else:
            raise NotImplementedError("No 2D disturbance!")


class ScreenTargetTracking(TargetTracking, Window):
    """Concrete implementation of Target Tracking task where the target is moving and
    are tracked by holding the cursor within the moving target"""

    limit2d = traits.Bool(True, desc="Limit cursor movement to 2D")
    limit1d = traits.Bool(True, desc="Limit cursor movement to 1D")

    sequence_generators = [
        'tracking_target_chain_1D', 'tracking_target_training'
    ]

    hidden_traits = ['cursor_color', 'trajectory_color', 'cursor_bounds', 'cursor_radius', 'plant_hide_rate', 'starting_pos']
    targets = []

    is_bmi_seed = True

    # Runtime settable traits
    target_radius = traits.Float(2, desc="Radius of targets in cm")
    trajectory_radius = traits.Float(.25, desc="Radius of targets in cm")
    trajectory_color = traits.OptionsList("gold", *target_colors, desc="Color of the trajectory", bmi3d_input_options=list(target_colors.keys()))
    target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    cursor_color = traits.OptionsList("pink", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    cursor_bounds = traits.Tuple((-10., 10., 0., 0., -10., 10.), desc='(x min, x max, y min, y max, z min, z max)')
    starting_pos = traits.Tuple((5., 0., 5.), desc='Where to initialize the cursor') 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant.set_bounds(np.array(self.cursor_bounds))
        self.plant.set_color(target_colors[self.cursor_color])
        self.plant.set_cursor_radius(self.cursor_radius)
        self.plant_vis_prev = True
        self.cursor_vis_prev = True
        
        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            # This is the target at the center being followed by the user
            self.target = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('trial', 'u4', (1,))
        self.add_dtype('plant_visible', '?', (1,))
        self.add_dtype('current_trajectory_coord', 'f8', (3,))
        super().init()
        self.plant.set_endpoint_pos(np.array(self.starting_pos))

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        prev_pt = np.copy(self.last_pt)
        self.move_effector()
        
        # Add Disturbance
        # Note: doesn't work with velocity control
        if self.disturbance_trial and self.state == "target": # If disturbance trial use disturbed position
            manual_pos = self.last_pt
            manual_vel = self.last_pt - prev_pt
            disturbance_position = self.add_disturbance(manual_pos, manual_vel,
                self.disturbance_path[self.frame_index],self.disturbance_path[self.frame_index-1])
            self.plant.set_endpoint_pos(disturbance_position)
            self.last_pt = self.plant.get_endpoint_pos()

        ## Run graphics commands to show/hide the plant if the visibility has changed
        self.update_plant_visibility()
        self.task_data['plant_visible'] = self.plant_visible

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        # Update the trial index
        self.task_data['trial'] = self.calc_trial_num()
        
        #Save the target position at each cycle. 
        if self.trial_timed_out or  self.trial_length == self.frame_index:
             self.task_data['current_trajectory_coord'] = [0,0,0]
        else:
            self.task_data['current_trajectory_coord'] = self.targs[self.frame_index]

        super()._cycle()

    def move_effector(self):
        '''Move the end effector, if a robot or similar is bself.baseline.show()eing controlled'''
        pass

    def run(self):
        '''
        See experiment.Experiment.run for documentation.
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        
        # Include some cleanup in case the parent class has errors
        try:
            super().run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_plant_visibility(self):
        ''' Update plant visibility'''
        if self.plant_visible != self.plant_vis_prev:
            self.plant_vis_prev = self.plant_visible
            self.plant.set_visibility(self.plant_visible)

    #### STATE FUNCTIONS ####
    def _start_wait(self):
        super()._start_wait()
        if self.calc_trial_num() == 0:
            self.add_model(self.target.graphics_models[0])
            self.target.hide()

        # Set up the next trajectory
        mytrajectory = np.array(np.squeeze(self.targs)[:,2])
        self.trajectory = VirtualCableTarget(target_radius=self.trajectory_radius, target_color=target_colors[self.trajectory_color],trajectory=mytrajectory)
        self.trial_length = np.shape(self.targs[:,2])[0]
        for model in self.trajectory.graphics_models:
            self.add_model(model)

        self.target.hide()
        self.trajectory.hide()

    def _start_initiation(self):
        super()._start_initiation()
        self.target.show()
        self.trajectory.show()
            
    def _start_tracking(self):
        super()._start_tracking()
        self.sync_event('TRIAL_START')
    
    def _start_reward(self):
        self.target.hide()
        self.target.cue_trial_end_success()
        self.sync_event('REWARD')

    def _end_reward(self):
        super()._end_reward()
        self.sync_event('TRIAL_END')
        self.trajectory.hide()
        self.target.hide()
        self.target.reset()

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        self.sync_event('OTHER_PENALTY')
        self.trajectory.hide()
        self.target.hide()
        self.target.reset()


    @staticmethod
    def calc_sum_of_sines(times, frequencies, amplitudes, phase_shifts):
        '''
        Generates the trajectories for the experiment
        '''
        t = times
        t = np.asarray(t).copy(); t.shape = (t.size,1)

        f = frequencies
        f = f.copy()
        f = np.reshape(f, (1,f.size))

        a = amplitudes
        a = a.copy()
        a = np.reshape(a, (1,a.size))

        p = phase_shifts        
        p = p.copy()
        p = np.reshape(p, (1,p.size))

        assert f.shape == a.shape == p.shape,"Shape of frequencies, amplitudes, and phase shifts must match"

        o = np.ones(t.shape)
        trajectory = np.sum(np.dot(o,a) * np.sin(2*np.pi*np.dot(t,f) + np.dot(o,p)),axis=1)
        
        return trajectory

    @staticmethod
    def calc_sum_of_sines_ramp(times, ramp, frequencies, amplitudes, phase_shifts):
            '''
            Adds a 1/t ramp up and ramp down at the start and end so the trajectories start and end at zero.
            '''
            t = times
            t = np.asarray(t).copy(); t.shape = (t.size,1)

            r = ramp

            trajectory = ScreenTargetTracking.calc_sum_of_sines(t, frequencies, amplitudes, phase_shifts)

            if r > 0:
                trajectory *= ((t*(t <= r)/r + (t > r)).flatten())**2
                #(((t*(t <= r)/r) + ((t > r) & (t < (t[-1]-r))) + ((t[-1]-t)*(t >= (t[-1]-r))/r)).flatten())**2

            return trajectory

    @staticmethod
    def generate_trajectories(num_trials=2, time_length=5, sample_rate=60, base_period=20, ramp=0):
        '''
        Sets up variables and uses prime numbers to call the above functions and generate then trajectories
        ramp is time length for preparatory lines
        '''
        hz = sample_rate # Hz -- sampling rate
        dt = 1/hz # sec -- sampling period

        T0 = base_period # sec -- base period
        w0 = 1/T0 # Hz -- base frequency

        r = ramp # "ramp" duration (see sum_of_sines_ramp)
        P = time_length/T0 # number of periods in signal
        T = time_length # sec -- signal duration
        dw = 1/T # Hz -- frequency resolution
        W = dw*T/dt/2 # Hz -- signal bandwidth

        full_primes = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
            101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199])
        primes_ind = np.where(full_primes <= T0)
        primes = full_primes[primes_ind]

        f = primes*w0 # stimulated frequencies
        f_ref = f.copy()
        f_dis = f.copy()

        a = 1/(1+np.arange(f.size)) # amplitude
        a_ref = a.copy()
        a_dis = a.copy()

        o = np.random.rand(primes.size) # phase offset
        o_ref = o.copy()
        o_dis = o.copy()

        t = np.arange(0,T,dt) # time samples
        w = np.arange(0,W,dw) # frequency samples

        N = t.size # = T/dt -- number of samples
        
        # create trials dictionary
        trials = dict(id=np.arange(num_trials), times=t, ref=np.zeros((num_trials,N)), dis=np.zeros((num_trials,N)))

        # randomize order of first two trials to provide random starting point
        order = np.random.choice([0,1])
        if order == 0:
            trial_order = [(1,'E','O'),(1,'O','E')]
        elif order == 1:
            trial_order = [(1,'O','E'),(1,'E','O')]

        # generate reference and disturbance trajectories for all trials
        for trial_id, (num_reps,ref_ind,dis_ind) in enumerate( functiontrial_order*int(num_trials/2)):   
            if ref_ind == 'E': 
                sines_r = np.arange(len(primes))[0::2] # use even indices
            elif ref_ind == 'O': 
                sines_r = np.arange(len(primes))[1::2] # use odd indices
            else:
                sines_r = np.arange(len(primes))
            if dis_ind == 'E':
                sines_d = np.arange(len(primes))[0::2]
            elif dis_ind == 'O':
                sines_d = np.arange(len(primes))[1::2]
            else:
                sines_d = np.arange(len(primes))
            
            ref_trajectory = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, f_ref[sines_r], a_ref[sines_r], o_ref[trial_id][sines_r])
            dis_trajectory = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, f_dis[sines_d], a_dis[sines_d], o_dis[trial_id][sines_d])
            
            # normalized trajectories
            trials['ref'][trial_id] = ref_trajectory/np.sum(a_ref)
            trials['dis'][trial_id] = dis_trajectory/np.sum(a_dis)
        
        return trials, trial_order

    ### Generator functions ####
    @staticmethod
    def tracking_target_debug(nblocks=1, ntrials=2, time_length=5, frequencies = [1,.75], boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis) target trajectories for debugging

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
        frequencies: numpy.ndarray
            A list of frequencies used to generate the trajectories
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0
        buffer_space_bef = int(60*1.3) # 1.3 seconds of straight line before trial #78 frames
        buffer_space_aft = int(60*1.5) # 1.5 seconds of straight line before trial #90 frames
        frames = int(np.round(time_length*60)) #300 frames when time_length=5
        y_primes_freq = np.array(frequencies)

        for i in range(nblocks):
            for j in range(ntrials):
                
                disturbance = False
                disturbance_path = np.zeros((frames+buffer_space_bef+buffer_space_aft,1))
                trajectory = np.zeros((frames+buffer_space_bef+buffer_space_aft,3))
                sum_of_sins_path = ScreenTargetTracking.generate_trajectory(y_primes_freq,20) # TODO
                pts = []
                # cannot broadcast array of length 768 to 468
                trajectory[:,2] = 5*np.concatenate((sum_of_sins_path[0]*np.ones(buffer_space_bef),sum_of_sins_path,sum_of_sins_path[-1]*np.ones(buffer_space_aft)))
                pts.append(trajectory)
                yield idx, pts, disturbance, disturbance_path
                idx += 1

    @staticmethod
    def tracking_target_chain_1D(nblocks=1, ntrials=2, time_length = 5, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis (vertical axis)) target trajectories

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
            The length of one target tracking trial in seconds 
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0
        buffer_space = int(60*1.5) #1.5 seconds of straight line before and after trial
        full_primes = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199])
        primes_ind = np.where(full_primes <= time_length)
        primes = full_primes[primes_ind]
        frames = int(np.round(time_length*60))
        disturbance_trials = np.random.randint(0,nblocks*ntrials,round(nblocks*ntrials*0.5))
        random_start = random.randint(0, 1)
        
        for i in range(nblocks):
            for j in range(ntrials):
                if idx % 2 == random_start:
                    y_primes_freq = primes[::2]
                    disturbance_freq = primes[::2]
                else:
                    y_primes_freq = primes[1::2]
                    disturbance_freq = primes[1::2]
                disturbance = False
                disturbance_path = np.zeros((frames+2*buffer_space,1))
                trajectory = np.zeros((frames+2*buffer_space,3))
                sum_of_sins_path = ScreenTargetTracking.generate_trajectory(y_primes_freq,time_length)
                pts = []
                trajectory[:,2] = 5*np.concatenate((np.zeros(buffer_space),sum_of_sins_path,np.zeros(buffer_space)))
                if np.any(idx == disturbance_trials):
                    disterb = ScreenTargetTracking.generate_trajectory(disturbance_freq,time_length,0.75)
                    disturbance_path = 5*np.concatenate((np.zeros(buffer_space),disterb,np.zeros(buffer_space)))
                    disturbance = True
                pts.append(trajectory)
                yield idx, pts, disturbance, disturbance_path
                idx += 1
    
    @staticmethod
    def tracking_target_training(nblocks=1, ntrials=2, time_length=5, frequencies = [1,.75], boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis) target trajectories for training

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
        frequencies: numpy.ndarray
            A list of frequencies used to generate the trajectories
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0
        buffer_space_bef = int(60*1.3) # 1.3 seconds of straight line before trial #78 frames
        buffer_space_aft = int(60*1.5) # 1.5 seconds of straight line before trial #90 frames
        frames = int(np.round(time_length*60)) #300 frames when time_length=5
        y_primes_freq = np.array(frequencies)

        for i in range(nblocks):
            for j in range(ntrials):
                
                disturbance = False
                disturbance_path = np.zeros((frames+buffer_space_bef+buffer_space_aft,1))
                trajectory = np.zeros((frames+buffer_space_bef+buffer_space_aft,3))
                sum_of_sins_path = ScreenTargetTracking.generate_trajectory(y_primes_freq,20)
                pts = []
                # cannot broadcast array of length 768 to 468
                trajectory[:,2] = 5*np.concatenate((sum_of_sins_path[0]*np.ones(buffer_space_bef),sum_of_sins_path,sum_of_sins_path[-1]*np.ones(buffer_space_aft)))
                pts.append(trajectory)
                yield idx, pts, disturbance, disturbance_path
                idx += 1