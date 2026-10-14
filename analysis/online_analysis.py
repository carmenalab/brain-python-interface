import datetime
import time
import json
import socket
import select
import os
import multiprocessing as mp
import threading
import queue
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, Slider, Cursor, TextBox
import aopy
import traits
from traits.trait_types import self
#from built_in_tasks.manualcontrolmultitasks import SaccadeTaskWithBackgroundTargets
from riglib.ecube import MultiSource, map_channels_for_multisource
from riglib.source import MultiChanDataSource

class OnlineDataWorker(threading.Thread):
    '''
    Worker process to receive data from BMI3D.
    '''

    def __init__(self, socket, result_queue):
        '''
        Start the server on the specified IP address and port number.
        '''
        self.sock = socket
        self._stop_event = threading.Event()
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        while not self._stop_event.is_set():
            ready = select.select([self.sock], [], [], 0.1) # 100ms timeout to check _stop_event
            if ready[0]:
                data = self.sock.recv(4096)
                key, value = data.decode('utf-8').split('%')
                self.result_queue.put((key, [json.loads(v) for v in value.split('#')]))

    def stop(self):
        self._stop_event.set()

class AnalysisWorker(mp.Process):
    '''
    Plots eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target.
    '''

    def __init__(self, task_params, data_queue, figsize=(8,8), update_rate=60, fps=60):
        self.task_params = task_params
        self._stop_event = mp.Event()
        self.data_queue = data_queue
        self.figsize = figsize
        self.update_rate = update_rate
        self.fps = fps
        super().__init__()

    def init(self):
        '''
        Initialize the worker. 
        '''
        self.cycle_count = 0

    def handle_data(self, key, values):
        '''
        Do something with incoming data. By default just keeps track of the time
        '''
        if key == 'cycle_count':
            self.cycle_count = values[0]

    def draw(self):
        '''
        Update the figure.
        '''
        if 'fps' in self.task_params:
            self.time_text.set_text(f"t={int(self.cycle_count/self.task_params['fps'])} s")
        else:
            self.time_text.set_text(f"t={int(self.cycle_count)} cycles")

    def cleanup(self):
        '''
        Cleanup tasks after the experiment ends, e.g. saving the figure.
        '''
        pass

    def update(self):
        while True:
            try:
                key, values = self.data_queue.get(timeout=0.) # continue if no data
                self.handle_data(key, values)
            except queue.Empty:
                break

    def run(self):
        print('Starting analysis worker:', self.__class__.__name__)
        
        # Initialize figure
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_axes([0, 0.2, 1, 0.7])
        self.ax.axis('off')
        experiment_name = self.task_params.get('experiment_name', 'None')
        te_id = self.task_params.get('te_id', 'None')
        self.fig.canvas.manager.set_window_title(f"{experiment_name} ({te_id}) - {self.__class__.__name__}")
        self.time_text = self.ax.text(1., 1.05, '', ha='right', va='center', fontsize=12, transform=self.ax.transAxes)
        self.time_text.set_text('Waiting for data...')
        self.init()
                
        # Pop up the figure
        plt.show(block=False)
        plt.pause(0.1)

        t_update = time.perf_counter()
        while plt.fignum_exists(self.fig.number) and (not self._stop_event.is_set()):
            if time.perf_counter() - t_update > 1./self.update_rate:
                try:
                    self.update()
                except:
                    traceback.print_exc()
                t_update = time.perf_counter()
            self.draw()
            self.time_text.set_text(f"t={int(self.cycle_count/self.task_params['fps'])}s")
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(1./self.fps) 

        self.cleanup()
        plt.close(self.fig)
        print(self.__class__.__name__, 'stopped')

    def stop(self):
        self._stop_event.set()


class BehaviorAnalysisWorker(AnalysisWorker):
    '''
    Plots eye, cursor, and target data from experiments that have them. Performs automatic
    calibration of eye data to target locations when the cursor enters the target if no
    calibration coefficients are available. 
    '''
   
    def __init__(self, task_params, data_queue, calibration_dir='/var/tmp', buffer_time=5, **kwargs):
        super().__init__(task_params, data_queue, **kwargs)
        self.calibration_dir = calibration_dir
        self.buffer_time = buffer_time

    def init(self):
        super().init()
        self.cursor_pos = np.zeros(2)
        self.eye_pos = np.zeros(2)
        self.calibrated_eye_pos = np.zeros(2)
        self.target_pos = {}
        self.targets = {}
        self.calibration_data = []
        self.calibration_flag = True
        self.eye_coeff = np.array([[1,0],[1,0]])
        self.eye_coeff_corr = 0.5 # Don't accept anything lower than 0.5 by default
        
        self.eye_buffer = np.zeros((int(self.buffer_time*self.task_params['fps']), 3))

        # Load previous calibration if it exists
        subject = self.task_params.get('subject_name', 'None')
        self.calibration_filename = f'calib_{subject}_{datetime.date.today()}.pkl'
        filepath = os.path.join(self.calibration_dir, self.calibration_filename)
        if os.path.exists(filepath):
            self.eye_coeff, self.eye_coeff_corr = aopy.data.pkl_read(self.calibration_filename, self.calibration_dir)
            self.calibration_flag = False

        # Turn off automatic calibration if this isn't a MC task
        if 'target_radius' not in self.task_params or self.task_params['target_radius'] >= 10:
            self.calibration_flag = False
        if 'rotation' not in self.task_params or self.task_params['rotation'] != 'yzx':
            self.calibration_flag = False

        # Set up figure
        bounds = self.task_params.get('cursor_bounds', (-10,10,0,0,-10,10))
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[-2], bounds[-1])
        self.ax.set_aspect('equal')
        self.circles = PatchCollection([])
        self.ax.add_collection(self.circles)

        # Set up eye diameter figure  
        self.eye_ax = self.fig.add_axes([0.1, 0.06, 0.8, 0.11])
        self.eye_ax.set_xlabel('Time (s)')
        self.eye_ax.set_ylabel('Eye Pos (cm)', color='blue')
        self.diam_ax = self.eye_ax.twinx()
        self.diam_ax.set_ylabel('Eye Diameter (cm)', color='green')
        self.diam_ax.set_xlim(-self.buffer_time, 0)
        self.diam_plot = self.diam_ax.plot([], [], 'green', linewidth=0.5, alpha=0.5)[0]
        self.x_plot = self.eye_ax.plot([], [], 'blue', linewidth=0.5, alpha=0.5)[0]
        self.y_plot = self.eye_ax.plot([], [], 'cyan', linewidth=0.5, alpha=0.5)[0]


    def update_eye_calibration(self):
        '''
        Update the eye calibration coefficients using the collected data
        '''
        if self.calibration_flag and len(self.calibration_data) > 3:
            eye_data, cursor_data = zip(*self.calibration_data)
            eye_data = np.array(eye_data)
            cursor_data = np.array(cursor_data)
            slopes, intercepts, correlation_coeff = aopy.analysis.fit_linear_regression(eye_data, cursor_data)
            if np.all(abs(correlation_coeff) > self.eye_coeff_corr):
                self.eye_coeff_corr = np.min(abs(correlation_coeff))
                self.eye_coeff = np.vstack((slopes, intercepts)).T
                print('updated calibration:', self.eye_coeff, self.eye_coeff_corr)
            
    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position and diameters
            targets (list): List of active targets in (position, radius, color) format
        '''
        calibrated_eye_pos = aopy.data.get_calibrated_eye_data(self.eye_pos[0:2], self.eye_coeff)
        try:
            radius = self.task_params['target_radius']
            color = self.task_params['target_color']
            targets = [(self.target_pos[k], radius, color if v == 1 else 'green') for k, v in self.targets.items() if v]
        except:
            targets = []
        return self.cursor_pos, calibrated_eye_pos, targets

    def handle_data(self, key, values):
        super().handle_data(key, values)
        if key == 'sync_event':
            event_name, event_data = values
            if event_name == 'TARGET_ON':
                self.targets[event_data] = 1
            elif event_name == 'TARGET_OFF':
                self.targets[event_data] = 0
            elif event_name in ['PAUSE', 'TRIAL_END', 'HOLD_PENALTY', 'DELAY_PENALTY', 'TIMEOUT_PENALTY','FIXATION_PENALTY']:
                # Clear targets at the end of the trial
                self.targets = {}
            elif event_name == 'REWARD':
                # Set all active targets to reward
                for target_idx in self.targets.keys():
                    self.targets[target_idx] = 2 if self.targets[target_idx] else 0
            elif event_name == 'CURSOR_ENTER_TARGET' and event_data > 0:
                self.calibration_data.append((self.eye_pos, self.cursor_pos))
                self.update_eye_calibration()
        elif key == 'cursor':
            self.cursor_pos = np.array(values[0])[[0,2]]
        elif key == 'eye_pos':
            self.eye_pos = np.array(values[0])[:2]
            self.temp = np.array(values[0])[:2]
            self.eye_buffer[:,:2] = np.roll(self.eye_buffer[:,:2], -1, axis=0)
            self.eye_buffer[-1,:2] = self.temp

        elif key == 'eye_diam':
            self.temp = np.array(values[0])
            if (np.size(self.temp) == 0) or (self.task_params.get('keyboard_control', False)):
                return
            self.eye_buffer[:,2] = np.roll(self.eye_buffer[:,2], -1, axis=0)
            self.eye_buffer[-1,2] = self.temp[0]

        elif key == 'calibrated_eye_pos':
            self.calibrated_eye_pos = np.array(values[0])[:2]

        elif key == 'target_location':
            target_idx, target_location = values
            self.target_pos[int(target_idx)] = np.array(target_location)[[0,2]]

    def draw(self):
        super().draw()
        cursor_pos, eye_pos, targets = self.get_current_pos()
        cursor_radius = self.task_params.get('cursor_radius', 0.25)

        patches = [
            plt.Circle(cursor_pos, cursor_radius), 
            plt.Circle(eye_pos, 0.5)
        ] + [plt.Circle(pos, radius) for pos, radius, _ in targets]
        self.circles.set_paths(patches)
        colors = ['b', 'g'] + [c for _, _, c in targets]
        self.circles.set_facecolor(colors)
        self.circles.set_alpha(0.5)

        self._draw_eye()

    def _draw_eye(self):
        # Update raw eye data plot
        self.x_plot.set_data(np.arange(len(self.eye_buffer)) * 1/(int(self.task_params['fps'])) - self.buffer_time, 
                                self.eye_buffer[:, 0])
        self.y_plot.set_data(np.arange(len(self.eye_buffer)) * 1/(int(self.task_params['fps'])) - self.buffer_time, 
                                self.eye_buffer[:, 1])
        self.eye_ax.relim()
        self.eye_ax.autoscale_view()
        self.diam_plot.set_data(np.arange(len(self.eye_buffer)) * 1/(int(self.task_params['fps'])) - self.buffer_time, 
                                self.eye_buffer[:, 2])
        self.diam_ax.relim()
        self.diam_ax.autoscale_view()


    def cleanup(self):

        # Save the calibration if it was performed
        if self.calibration_flag and not np.array_equal(self.eye_coeff, np.array([[1,0],[1,0]])):
            aopy.data.pkl_write(self.calibration_filename, (self.eye_coeff, self.eye_coeff_corr), self.calibration_dir)

class SaccadeAnalysisWorker(BehaviorAnalysisWorker):
    '''
    Plots calibrated_eye, cursor, and target data from experiments that have them.
    This is for eye-related task that requires calibrated eye position
    '''
 
    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position and diameters
            targets (list): List of active targets in (position, radius, color) format
        '''
        try:
            radius = self.task_params['target_radius']
            color = 'orange'
            targets = [(self.target_pos[k], radius, color if v == 1 else 'green') for k, v in self.targets.items() if v]
        except:
            targets = []
        
        return self.cursor_pos, self.calibrated_eye_pos, targets

    def draw(self):
        super(BehaviorAnalysisWorker, self).draw()
        cursor_pos, calibrated_eye_pos, targets = self.get_current_pos()
        cursor_radius = self.task_params.get('cursor_radius', 0.25)
        if 'fixation_radius_buffer' in self.task_params:
            buffer = self.task_params['fixation_radius_buffer']
        elif 'fixation_dist' in self.task_params:
            buffer = self.task_params['fixation_dist'] - self.task_params['target_radius']
        else:
            buffer=0
        eye_radius = 0.1

        patches1 = [plt.Circle(pos, radius+buffer) for pos, radius, _ in targets]
        patches2 = [plt.Circle(cursor_pos, cursor_radius), plt.Circle(calibrated_eye_pos, eye_radius)]
        patches3 = [plt.Circle(pos, radius) for pos, radius, _ in targets]
        patches = patches1 + patches2 + patches3
        self.circles.set_paths(patches)
        colors = [[0.8,0.8,0.8] for _, _, c in targets]  + ['darkblue', 'darkgreen']  + [c for _, _, c in targets] 
        self.circles.set_facecolor(colors)
        self.circles.set_alpha(0.5)

        self._draw_eye()

class EyeHandAnalysisWorker(BehaviorAnalysisWorker):
    '''
    Plots calibrated_eye, cursor, and target data from experiments that have them. 
    This is for eye-hand task
    '''

    def init(self):
        super().init()
        self.calibrated_eye_pos = np.zeros(2)
        self.hand_targets = {}
        self.eye_targets = {}
        self.target_pos = []
        self.target_idx_trial = []

    def handle_data(self, key, values):
        super(BehaviorAnalysisWorker, self).handle_data(key, values)
        if key == 'sync_event':
            event_name, event_data = values
            if event_name == 'TARGET_ON':
                self.hand_targets[event_data] = 1 # event data represents target index in bmi3d
                #self.eye_targets[self.target_idx_trial[0]] = 1 # bacause eye initial target and hand target appear at the same time
            elif event_name == 'TARGET_OFF':
                self.hand_targets[event_data] = 0
            elif event_name == 'EYE_TARGET_ON':
                self.eye_targets[event_data] = 1
            elif event_name == 'EYE_TARGET_OFF':
                self.eye_targets[event_data] = 0

                if self.task_params['experiment_name'] == 'EyeConstrainedReachingTask':
                    self.hand_targets[self.target_idx_trial[-1]] = 0 # In this task, the hand target also disappear

            elif event_name in ['PAUSE', 'TRIAL_END', 'HOLD_PENALTY', 'DELAY_PENALTY', 'TIMEOUT_PENALTY','FIXATION_PENALTY','OTHER_PENALTY']:
                # Clear targets at the end of the trial
                self.hand_targets = {}
                self.eye_targets = {}
                self.target_pos = []
                self.target_idx_trial = []

            elif event_name == 'REWARD':
                # Set all active targets to reward
                for target_idx in self.hand_targets.keys():
                    self.hand_targets[target_idx] = 2 if self.hand_targets[target_idx] else 0
                for target_idx in self.eye_targets.keys():
                    self.eye_targets[target_idx] = 2 if self.eye_targets[target_idx] else 0

        elif key == 'cursor':
            self.cursor_pos = np.array(values[0])[[0,2]]
        elif key == 'calibrated_eye_pos':
            self.calibrated_eye_pos = np.array(values[0])[:2]

        elif key == 'eye_pos':
            self.eye_pos = np.array(values[0])[:2]
            self.temp = np.array(values[0])[:2]
            self.eye_buffer[:,:2] = np.roll(self.eye_buffer[:,:2], -1, axis=0)
            self.eye_buffer[-1,:2] = self.temp

        elif key == 'eye_diam':
            self.temp = np.array(values[0])[0]
            self.eye_buffer[:,2] = np.roll(self.eye_buffer[:,2], -1, axis=0)
            self.eye_buffer[-1,2] = self.temp

        elif key == 'target_location':
            target_idx, target_location = values
            self.target_pos.append(np.array(target_location)[[0,2]])
            self.target_idx_trial.append(target_idx)

            
    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position and diameters
            targets (list): List of active targets in (position, radius, color) format
        '''
        try:
            radius = self.task_params['target_radius']
            eye_radius = self.task_params['fixation_radius']
            color = 'orange'
            eye_color = 'lightskyblue'
            eye_targets = [(self.target_pos[0], eye_radius, eye_color if v == 1 else 'green') for k, v in self.eye_targets.items() if v and k < 3]
            eye_targets.extend([(self.target_pos[1], eye_radius, eye_color if v == 1 else 'green') for k, v in self.eye_targets.items() if v and k >= 3])
            hand_targets = [(self.target_pos[2], radius, color if v == 1 else 'green') for k, v in self.hand_targets.items() if v]
        except:
            eye_targets = []
            hand_targets = []
        
        return self.cursor_pos, self.calibrated_eye_pos, eye_targets, hand_targets

    def draw(self):
        super(BehaviorAnalysisWorker, self).draw()
        cursor_pos, calibrated_eye_pos, eye_targets, hand_targets = self.get_current_pos()
        cursor_radius = self.task_params.get('cursor_radius', 0.25)
        if 'fixation_radius_buffer' in self.task_params:
            buffer = self.task_params['fixation_radius_buffer']
        elif 'fixation_dist' in self.task_params:
            buffer = self.task_params['fixation_dist'] - self.task_params['target_radius']
        eye_radius = 0.2

        patches1 = [plt.Circle(pos, radius+buffer) for pos, radius, _ in eye_targets]
        patches2 = [plt.Circle(cursor_pos, cursor_radius), plt.Circle(calibrated_eye_pos, eye_radius)]
        patches3 = [plt.Circle(pos, radius) for pos, radius, _ in eye_targets]
        patches4 = [plt.Circle(pos, radius) for pos, radius, _ in hand_targets]
        patches = patches1 + patches2 + patches3 + patches4
        self.circles.set_paths(patches)
        colors = [[0.8,0.8,0.8] for _, _, c in eye_targets]  + ['darkblue', 'darkgreen']  + [c for _, _, c in eye_targets] + [c for _, _, c in hand_targets]
        self.circles.set_facecolor(colors)
        self.circles.set_alpha(0.5)

        self._draw_eye()

class EyeHandSequenceAnalysisWorker(BehaviorAnalysisWorker):
    '''
    Plots calibrated_eye, cursor, and target data from experiments that have them. 
    This is for eye-hand task
    '''

    def init(self):
        super().init()
        self.calibrated_eye_pos = np.zeros(2)
        self.hand_targets = {}
        self.eye_targets = {}
        self.is_sequence = False

    def handle_data(self, key, values):
        super(BehaviorAnalysisWorker, self).handle_data(key, values)
        if key == 'cycle_count':
            self.cycle_count = values[0]       
        elif key == 'sync_event':
            event_name, event_data = values
            if event_name == 'TARGET_ON':
                self.hand_targets[event_data] = 1
            elif event_name == 'TARGET_OFF':
                self.hand_targets[event_data] = 0
            elif event_name == 'EYE_TARGET_ON':
                self.hand_targets[event_data] = 1
                self.eye_targets[event_data] = 1
            elif event_name == 'EYE_TARGET_OFF':
                if not self.is_sequence:
                    self.hand_targets[event_data] = 0

                self.eye_targets[event_data] = 0
            elif event_name in ['PAUSE', 'TRIAL_END', 'HOLD_PENALTY', 'DELAY_PENALTY', 'TIMEOUT_PENALTY','FIXATION_PENALTY', 'OTHER_PENALTY']:
                # Clear targets at the end of the trial
                self.hand_targets = {}
                self.eye_targets = {}
                self.is_sequence = False

            elif event_name == 'REWARD':
                # Set all active targets to reward
                for target_idx in self.hand_targets.keys():
                    self.hand_targets[target_idx] = 2 if self.hand_targets[target_idx] else 0
                for target_idx in self.eye_targets.keys():
                    self.eye_targets[target_idx] = 2 if self.eye_targets[target_idx] else 0

        elif key == 'cursor':
            self.cursor_pos = np.array(values[0])[[0,2]]
        elif key == 'eye_pos':
            self.eye_pos = np.array(values[0])[:2]
            self.temp = np.array(values[0])[:2]
            self.eye_buffer[:,:2] = np.roll(self.eye_buffer[:,:2], -1, axis=0)
            self.eye_buffer[-1,:2] = self.temp

        elif key == 'eye_diam':
            self.temp = np.array(values[0])[0]
            self.eye_buffer[:,2] = np.roll(self.eye_buffer[:,2], -1, axis=0)
            self.eye_buffer[-1,2] = self.temp

        elif key == 'calibrated_eye_pos':
            self.calibrated_eye_pos = np.array(values[0])[:2]

        elif key == 'target_location':
            target_idx, target_location = values
            self.target_pos[int(target_idx)] = np.array(target_location)[[0,2]]

        elif key == 'is_sequence':
            self.is_sequence = values[0]


    def get_current_pos(self):
        '''
        Get the current cursor, eye, and target positions

        Returns:
            cursor_pos ((2,) tuple): Current cursor position
            eye_pos ((2,) tuple): Current eye position and diameters
            targets (list): List of active targets in (position, radius, color) format
        '''
        try:
            radius = self.task_params['target_radius']
            eye_radius = self.task_params['fixation_radius']
            color = 'orange'
            eye_color = 'darkcyan'
            eye_targets = [(self.target_pos[k], eye_radius, eye_color if v == 1 else 'green') for k, v in self.eye_targets.items() if v]
            hand_targets = [(self.target_pos[k], radius, color if v == 1 else 'green') for k, v in self.hand_targets.items() if v]
        except:
            eye_targets = []
            hand_targets = []
        
        return self.cursor_pos, self.calibrated_eye_pos, eye_targets, hand_targets

    def draw(self):
        super(BehaviorAnalysisWorker, self).draw()
        cursor_pos, calibrated_eye_pos, eye_targets, hand_targets = self.get_current_pos()
        cursor_radius = self.task_params.get('cursor_radius', 0.25)
        if 'fixation_radius_buffer' in self.task_params:
            buffer = self.task_params['fixation_radius_buffer']
        elif 'fixation_dist' in self.task_params:
            buffer = self.task_params['fixation_dist'] - self.task_params['target_radius']
        eye_radius = 0.2

        patches1 = [plt.Circle(pos, radius+buffer) for pos, radius, _ in eye_targets]
        patches2 = [plt.Circle(pos, radius) for pos, radius, _ in hand_targets]
        patches3 = [plt.Circle(pos, radius) for pos, radius, _ in eye_targets]
        patches4 = [plt.Circle(cursor_pos, cursor_radius), plt.Circle(calibrated_eye_pos, eye_radius)]
        
        patches = patches1 + patches2 + patches3 + patches4
        self.circles.set_paths(patches)
        colors = [[0.8,0.8,0.8] for _, _, c in eye_targets]  + [c for _, _, c in hand_targets] + [c for _, _, c in eye_targets] +  ['indigo', 'darkgreen']  
        self.circles.set_facecolor(colors)
        self.circles.set_alpha(0.7)

        self._draw_eye()


class ERPAnalysisWorker(AnalysisWorker):
    '''
    Plots ERP data from experiments with an ECoG244 array. Automatically calculates 
    ERPs for flash, movement, or laser events depending on the task.
    '''
    bufferlen = 5 # seconds of data to keep in the buffer

    def __init__(self, task_params, data_queue, update_rate=1, time_before=0.1, 
                 time_after=0.1, figure_dir='/home/aolab/figures',
                 condition_idx=None, stim_sites=None, title=None, **kwargs):
        super().__init__(task_params, data_queue, update_rate=update_rate, **kwargs)
        self.time_before = time_before
        self.time_after = time_after
        self.figure_dir = figure_dir
        self.title = title
        try:
            self.condition_idx = [int(condition_idx)]
        except:
            self.condition_idx = condition_idx
        self.stim_sites = stim_sites
        if self.condition_idx is not None and len(self.stim_sites) == 0:
            raise ValueError('Condition index provided but no stim sites were present in the experiment')
        self.current_condition_idx = 0

    def init(self):
        super().init()

        # Initialize the data source
        self.elec_pos, self.acq_ch, _ = aopy.data.load_chmap('ECoG244')
        self.clock_dch = self.task_params.get('screen_sync_dch', 40)
        self.headstage = self.task_params.get('headstage_connector', 7)
        self.clock_elapsed = 0
        self.clock_times = np.array([])
        self.trigger_dch = None
        self.trigger_events = []
        self.trigger_cycles = []
        self.trigger_times = []
        self.lfp_downsample = 25
        self.erp_since_update = 0

        # Determine the trigger channel and events based on the task parameters
        if 'qwalor_trigger_dch' in self.task_params:
            self.trigger_dch = self.task_params['qwalor_trigger_dch']
            channels = map_channels_for_multisource(headstage_channels=self.acq_ch, 
                                                         digital_channels=[self.clock_dch, self.trigger_dch])
            print('Triggering on qwalor channel', self.trigger_dch)
            print('Using clock channel', self.clock_dch)
        elif 'qwalor_ch1_enable' in self.task_params: # TODO: support multiple lasers
            for idx in range(1,5):
                if self.task_params[f'qwalor_ch{idx}_enable']:
                    self.trigger_dch = self.task_params[f'qwalor_ch{idx}_trigger_dch']
                    break
            channels = map_channels_for_multisource(headstage_channels=self.acq_ch,
                                                            digital_channels=[self.clock_dch, self.trigger_dch])
            print('Triggering on qwalor channel', self.trigger_dch)
            print('Using clock channel', self.clock_dch)
        else:
            self.trigger_events.append('TARGET_ON') # For flash
            channels = map_channels_for_multisource(headstage_channels=self.acq_ch, digital_channels=[self.clock_dch])
            print('Triggering on events', self.trigger_events)
            print('Using clock channel', self.clock_dch)
        self.ds = MultiChanDataSource(MultiSource, channels=channels, bufferlen=self.bufferlen, headstage=self.headstage)
        self.ds.start()
        print('datasource started')

        # Initialize the ERP data
        self.erp = np.zeros((int((self.time_before + self.time_after) * self.ds.source.update_freq/self.lfp_downsample), 
                             len(self.elec_pos), 0), dtype=self.ds.source.dtype)
        
        # Initialize the figure
        self.data_map = np.zeros((16,16))
        self.erp_im = aopy.visualization.plot_spatial_map(self.data_map, self.elec_pos[:,0], self.elec_pos[:,1], 
                                                 cmap='bwr', ax=self.ax)
        self.erp_im.set_clim(-100, 100)
        self.erp_text = self.ax.text(0.75, 1.05, '.', ha='center', va='center', fontsize=12, transform=self.ax.transAxes)
        if self.title is not None:
            self.ax.set_title(self.title)

        # Add a slider to control the range
        ax_slider = self.fig.add_axes([0.225, 0.075, 0.6, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Range (uV)',
            valmin=0.1,
            valmax=1000,
            valinit=100
        )
        self.slider.on_changed(lambda val: self.erp_im.set_clim(-val, val))

        # And a button to show/hide channel numbers
        self.labels = []
        elec_pos, acq_ch, _ = aopy.data.load_chmap('ECoG244')
        stim_pos, stim_ch, _ = aopy.data.load_chmap('Opto32')
        for pos, ch in zip(elec_pos, acq_ch):
            self.labels.append(aopy.visualization.annotate_spatial_map(pos, ch, 'c', fontsize=12, ax=self.ax))
        for pos, ch in zip(stim_pos, stim_ch):
            self.labels.append(aopy.visualization.annotate_spatial_map(pos, ch, 'm', fontsize=12, ax=self.ax))
        ax_toggle = self.fig.add_axes([0.1, 0.025, 0.1, 0.025])
        self.toggle = Button(ax_toggle, 'Labels')
        def toggle_labels(event):
            self.toggle.color = '0.85' if self.labels[0].get_visible() else '0.5'
            for label in self.labels:
                label.set_visible(not label.get_visible())
        toggle_labels(None)
        self.toggle.on_clicked(toggle_labels)

        # Add a toggle between voltage and z-score
        self.zscore = False
        def toggle_zscore(event):
            self.zscore = not self.zscore
            if self.zscore:
                self.slider.valmax = 10
                self.slider.set_val(5)
                self.zscore_button.color = '0.5'
            else:
                self.slider.valmax = 1000
                self.slider.set_val(100)
                self.zscore_button.color = '0.85'
            self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
            self._update_erp_data()
            self.draw()

        self.ax_zscore = self.fig.add_axes([0.225, 0.025, 0.1, 0.025])
        self.zscore_button = Button(self.ax_zscore, 'Z-score')
        self.zscore_button.on_clicked(toggle_zscore)

    def update(self):
        super().update()
        # Keep track of clock cycles
        clock_data = self.ds.get_new(map_channels_for_multisource(digital_channels=[self.clock_dch]))[0]
        if len(clock_data) == 0:
            return # no new data
        timestamps, edges = aopy.utils.detect_edges(clock_data, self.ds.source.update_freq, rising=True, falling=False)
        self.clock_times = np.concatenate((self.clock_times, timestamps + self.clock_elapsed))
        clock_elapsed_prev = self.clock_elapsed
        clock_elapsed_new = self.clock_elapsed + len(clock_data) / self.ds.source.update_freq

        # Check the trigger if it exists
        if self.trigger_dch:
            trigger_data = self.ds.get_new(map_channels_for_multisource(digital_channels=[self.trigger_dch]))[0]
            timestamps, edges = aopy.utils.detect_edges(trigger_data, self.ds.source.update_freq, rising=True, falling=False)   
            self.trigger_times = np.concatenate((self.trigger_times, timestamps + clock_elapsed_prev)).tolist()

        # Only count triggers that occur during the specified condition if a condition index was provided
        in_condition = True
        if self.condition_idx is not None:
            in_condition = self.current_condition_idx in self.condition_idx

        # Append new ERPs from trigger events
        fs = self.ds.source.update_freq / self.lfp_downsample
        nt = 2./self.update_rate # seconds
        npts = int(nt * self.ds.source.update_freq)
        lfp = self.ds.get(npts, map_channels_for_multisource(headstage_channels=self.acq_ch))
        lfp = np.array(lfp)[:,::self.lfp_downsample].T # reshape and downsample (nt, nch)
        erp_count = 0
        ignored_trigger_cycles = []
        while len(self.trigger_cycles) > 0:
            cycle = self.trigger_cycles.pop()
            if cycle >= len(self.clock_times):
                ignored_trigger_cycles.append(cycle) # future cycle we haven't seen yet, check back on the next update
                continue
            time = self.clock_times[cycle]
            if time + self.time_after > clock_elapsed_new:
                ignored_trigger_cycles.append(cycle) # future trigger we haven't seen yet, check back on the next update
                continue
            elif time - self.time_before < clock_elapsed_new - nt:
                print('missed cycle', cycle, 'at time', time) # missed trigger we don't have any data for, just skip it
                continue
            if not in_condition:
                continue # trigger occurred outside of the specified condition, ignore it
            erp = aopy.analysis.calc_erp(lfp, [time - clock_elapsed_new + nt], self.time_before, self.time_after, fs)
            self.erp = np.concatenate((self.erp, erp), axis=2)
            erp_count += 1
        self.trigger_cycles = ignored_trigger_cycles

        # Check for new digital triggers
        ignored_trigger_times = []
        while len(self.trigger_times) > 0:
            time = self.trigger_times.pop()
            if time + self.time_after > clock_elapsed_new:
                ignored_trigger_times.append(time) # future trigger we haven't seen yet, check back on the next update
                continue
            elif time - self.time_before < clock_elapsed_new - nt:
                print('missed trigger', time) # missed trigger we don't have any data for, just skip it
                continue
            if not in_condition:
                continue # trigger occurred outside of the specified condition, ignore it
            erp = aopy.analysis.calc_erp(lfp, [time - clock_elapsed_new + nt], self.time_before, self.time_after, fs)
            self.erp = np.concatenate((self.erp, erp), axis=2)
            erp_count += 1
        self.trigger_times = ignored_trigger_times

        # Update the total elapsed time
        self.clock_elapsed = clock_elapsed_new

        # Update the ERP
        self.erp_since_update += erp_count
        self._update_erp_data()

    def _update_erp_data(self):
        if self.erp_since_update == 0:
            return
        if (self.erp.shape[2] > 50) and (self.erp_since_update < 10):
            return  # only update the plot every 10 new ERPs to save on computation
        self.erp_since_update = 0
        
        fs = self.ds.source.update_freq / self.lfp_downsample
        if self.zscore:
            altcond_window = (0, self.time_after)
            nullcond_window = (-self.time_before, 0)
            altcond, nullcond = aopy.analysis.latency.prepare_erp(self.erp, self.erp, fs, 
                                                                self.time_before, self.time_after, 
                                                                nullcond_window, altcond_window)[:2]
            z_erp = np.std(nullcond, axis=0)
            sd_erp = (altcond - np.mean(nullcond, axis=0)) / z_erp
            max_erp = aopy.analysis.get_max_erp(sd_erp, 0, self.time_after, fs, trial_average=True)
        else:
            max_erp = aopy.analysis.get_max_erp(self.erp, self.time_before, self.time_after, fs, trial_average=True)
            max_erp *= 1.907348633e-7*1e6 # convert to microvolts
        self.data_map = aopy.visualization.get_data_map(max_erp, self.elec_pos[:,0], self.elec_pos[:,1])

    def handle_data(self, key, values):
        super().handle_data(key, values)
        if key == 'sync_event':
            event_name, event_data = values
            if event_name in self.trigger_events:
                self.trigger_cycles.append(self.cycle_count)
        elif key == 'laser_conditions': # multiple lasers on each trial
            self.current_condition_idx = values[0][0] # pick first one for now, TODO: support multiple conditions at once
        elif key == 'laser_condition': # single laser on each trial
            self.current_condition_idx = values[0]

    def draw(self):
        super().draw()
        self.erp_im.set_data(self.data_map)
        self.erp_text.set_text(f"{self.erp.shape[2]} trials")

    def cleanup(self):
        '''
        Cleanup tasks after the experiment ends, e.g. saving the figure.
        '''
        self.ds.stop()      
        subject = self.task_params.get('subject_name', 'None')
        te_id = self.task_params.get('te_id', 'None')
        if te_id == 'None':
            return
        date = datetime.date.today()
        if self.title is not None:
            filename = f'online_erp_{self.title.replace(" ", "_")}_{subject}_{te_id}_{date}.png'
        else:
            filename = f'online_erp_{subject}_{te_id}_{date}.png'
        plt.figure(self.fig)
        aopy.visualization.savefig(self.figure_dir, filename, transparent=False)

class SLICAnalysisWorker(ERPAnalysisWorker):
    '''
    Plots SLIC connectivity from experiments with an ECoG244 array recordings.
    '''
    def init(self):
        super().init()
        self.erp_im.set_clim(-0.1, 0.1)
        stim_site = self.stim_sites[self.condition_idx[0]]
        ch_near_stim = aopy.analysis.connectivity.get_acq_ch_near_stimulation_site(stim_site)
        _, acq_ch, _ = aopy.data.load_chmap('ECoG244')
        self.stim_ch = np.where(np.isin(acq_ch, ch_near_stim))[0]
        self.taper_len = 0.06
        self.angle_map = self.data_map.copy()
        self.slic_map = self.data_map.copy()

        # Add a toggle between coherence and phase
        del self.zscore_button # hide the z-score button
        self.show_angle = False
        self.slider.valmin = 0
        self.slider.valmax = 1
        self.slider.set_val(0.1)
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        def toggle_angle(event):
            self.show_angle = not self.show_angle
            if self.show_angle:
                self.erp_im.set_cmap('YlGnBu')
                self.erp_im.set_clim=(0, 2*np.pi),
                self.slider.drag_active = False # disable the slider for phase
            else:
                self.erp_im.set_cmap('bwr')
                self.erp_im.set_clim=(-self.slider.val, self.slider.val), 
                self.erp_im.set_data(self.data_map)
                self.slider.drag_active = True
            self.draw()

        self.ax_angle = self.fig.add_axes([0.225, 0.025, 0.1, 0.025])
        self.angle_button = Button(self.ax_angle, 'Phase')
        self.angle_button.on_clicked(toggle_angle)

        # Add text boxes to enter the frequency band of interest
        self.band = [80, 150]
        def on_lf_text_submit(text):
            try:
                lf = float(text)
                self.band[0] = lf
            except:
                pass
        def on_hf_text_submit(text):
            try:
                hf = float(text)
                self.band[1] = hf
            except:
                pass
        axlf = self.fig.add_axes([0.525, 0.025, 0.05, 0.025])
        self.lf_text = TextBox(axlf, 'Min freq (Hz)', initial='80')
        self.lf_text.on_submit(on_lf_text_submit)
        axhf = self.fig.add_axes([0.725, 0.025, 0.05, 0.025])
        self.hf_text = TextBox(axhf, 'Max freq (Hz)', initial='150')
        self.hf_text.on_submit(on_hf_text_submit)
        print('finished init')

    def _update_erp_data(self):
        if self.erp_since_update == 0:
            return
        if (self.erp_since_update < 10):
            return  # only update the plot every 10 new ERPs to save on computation
        self.erp_since_update = 0

        fs = self.ds.source.update_freq / self.lfp_downsample
        freqs, time, coh_all, angle_all = aopy.analysis.connectivity.calc_connectivity_map_coh(
            self.erp, fs, self.time_before, self.time_after, self.stim_ch, 
            window=(-self.taper_len,self.taper_len), n=self.taper_len, step=self.taper_len, 
            parallel=False, verbose=False
        )
        angle = aopy.analysis.calc_tfr_mean(freqs, time[1:], angle_all[:,[1],:], self.band)
        self.angle_map = aopy.visualization.get_data_map(angle, self.elec_pos[:,0], self.elec_pos[:,1])
        slic = aopy.analysis.calc_tfr_mean(freqs, time[1:], coh_all[:,[1],:]-coh_all[:,[0],:], self.band)
        self.slic_map = aopy.visualization.get_data_map(slic, self.elec_pos[:,0], self.elec_pos[:,1])

    def draw(self):
        if self.show_angle:
            self.data_map = self.angle_map
        else:
            self.data_map = self.slic_map
        super().draw()


class BMIAnalysisWorker(AnalysisWorker):
    '''
    Plots neural features and decoder state data from experiments that have a decoder. 
    '''
   
    def __init__(self, task_params, data_queue, buffer_time=10, **kwargs):
        super().__init__(task_params, data_queue, **kwargs)
        self.buffer_time = buffer_time
        
    def init(self):
        super().init()
        self.channels = self.task_params['decoder_channels']
        self.states = self.task_params['decoder_states']
        self.bands = self.task_params['decoder_bands']
        self.neural_feats = np.zeros((int(self.buffer_time*self.task_params['fps']), len(self.channels)*len(self.bands)))
        self.decoder_states = np.zeros((int(self.buffer_time*self.task_params['fps']), len(self.states)))

        # Create feature axes  
        self.feat_ax = self.fig.add_axes([0.1, 0.06, 0.8, 0.4])
        self.feat_ax.set_xlim(-self.buffer_time, 0)
        self.feat_ax.set_xlabel('Time (s)')
        self.feat_ax.set_ylabel('Neural features')
        time = np.arange(len(self.neural_feats)) * 1/(int(self.task_params['fps'])) - self.buffer_time
        self.feat_plots = self.feat_ax.plot(time, self.neural_feats)

        # Create state axes
        self.state_ax = self.fig.add_axes([0.1, 0.56, 0.8, 0.4])
        self.state_ax.set_xlim(-self.buffer_time, 0)
        self.state_ax.set_xlabel('Time (s)')
        self.state_ax.set_ylabel('Decoder state')
        self.state_plots = self.state_ax.plot(time, self.decoder_states)

    def handle_data(self, key, values):
        super().handle_data(key, values)
        if key == 'decoder_state':
            self.decoder_states = np.roll(self.decoder_states, -1, axis=0)
            self.decoder_states[-1] = np.array(values[0])
        elif key == 'neural_features':
            self.neural_feats = np.roll(self.neural_feats, -1, axis=0)
            self.neural_feats[-1] = np.array(values[0])

    def _set_axis_limits(self, axis, data):
        if data.size == 0:
            axis.set_ylim(-1, 1)
            return

        data_min = np.min(data)
        data_max = np.max(data)
        if data_min == data_max:
            pad = 1 if data_min == 0 else abs(data_min) * 0.1
            axis.set_ylim(data_min - pad, data_max + pad)
        else:
            axis.set_ylim(data_min, data_max)

    def draw(self):
        super().draw()
        time = np.arange(len(self.neural_feats)) * 1/(int(self.task_params['fps'])) - self.buffer_time
        for i, plot in enumerate(self.feat_plots):
            plot.set_data(time, self.neural_feats[:,i])
        self._set_axis_limits(self.feat_ax, self.neural_feats)
        for i, plot in enumerate(self.state_plots):
            plot.set_data(time, self.decoder_states[:,i])
        self._set_axis_limits(self.state_ax, self.decoder_states)

class OnlineDataServer(threading.Thread):
    '''
    Interface for accumulating and analyzing BMI3D data in real-time.
    '''

    def __init__(self, host_ip, port=5000):
        '''
        Initialize the server on the specified IP address and port number.

        Args:
            host_ip (str): IP address of the machine running the online analysis
            port (int): Port number for the online analysis server
        '''

        # Initialize socket
        self.sock = socket.socket(
            socket.AF_INET, # Internet
              socket.SOCK_DGRAM) # UDP 
        self.sock.bind((host_ip, port))
        self.sock.setblocking(0)
                
        # Initialize workers
        self.data_worker = None
        self.analysis_workers = []
        try:
            MultiSource.pre_init()
            print('eCube streaming initialized')
        except:
            print('Error initializing eCube streaming')
            traceback.print_exc()

        # Initialize the server
        self._stop_event = threading.Event()
        self.reset()
        self.is_completed = False
        super().__init__()

    def _stop(self):
        # Stop all the workers
        for worker, _ in self.analysis_workers:
            if worker.is_alive():
                worker.stop()
                worker.join()
        self.analysis_workers = []
        if self.data_worker and self.data_worker.is_alive():
            self.data_worker.stop()
            self.data_worker.join()
        self.data_worker = None

    def reset(self):
        self._stop()

        # Start new workers
        self.result_queue = queue.Queue()
        self.data_worker = OnlineDataWorker(self.sock, self.result_queue)
        self.data_worker.start()
        self.is_running = False
        self.is_completed = True
        self.task_params = {}
        self.state = None

    def init(self):
        '''
        Once the experiment is initialized but before it starts, we spin up the analysis processes
        based on what kind of experiment is running.
        '''
        # Open the behavior analysis worker
        print('init in state', self.state)
        data_queue = mp.Queue()
        if self.task_params['experiment_name'] == 'ManualControl':
            self.analysis_workers.append((BehaviorAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'EyeConstrainedManualControl':
            self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'SaccadeTask':
            self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))
        
        elif self.task_params['experiment_name'] == 'SaccadeTaskWithBackgroundTargets':
                    self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'FlashTargets':
            self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'HandConstrainedSaccadeTask':
            self.analysis_workers.append((EyeHandAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'EyeConstrainedReachingTask':
            self.analysis_workers.append((EyeHandAnalysisWorker(self.task_params, data_queue), data_queue))        

        elif self.task_params['experiment_name'] == 'EyeHandConstrainedReachingTask':
            self.analysis_workers.append((EyeHandAnalysisWorker(self.task_params, data_queue), data_queue))  

        elif self.task_params['experiment_name'] == 'EyeHandConstrainedSequentialReachingTask':
            self.analysis_workers.append((EyeHandAnalysisWorker(self.task_params, data_queue), data_queue))   

        elif self.task_params['experiment_name'] == 'EyeHandSequenceTask':
            self.analysis_workers.append((EyeHandSequenceAnalysisWorker(self.task_params, data_queue), data_queue))  

        elif self.task_params['experiment_name'] == 'TargetCaptureVisualFeedback':
            self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))

        elif self.task_params['experiment_name'] == 'TargetCaptureVisualFeedbackEyeConstrained':
            self.analysis_workers.append((SaccadeAnalysisWorker(self.task_params, data_queue), data_queue))   

        # Is there ecube neural data?
        if 'record_headstage' in self.task_params and self.task_params['record_headstage']:
            stim_site = self.task_params.get('stimulation_site', 0)
            try:
                stim_sites = [int(stim_site)]
            except:
                stim_sites = [int(s) for s in stim_site]
            for idx, site in enumerate(stim_sites):
                if site == -1:
                    continue
                title = f"Stim site {site}"
                data_queue = mp.Queue()
                self.analysis_workers.append((ERPAnalysisWorker(
                    self.task_params, data_queue, condition_idx=idx, stim_sites=stim_sites, 
                    title=title), data_queue))
                title = f"SLIC Stim site {site}"
                data_queue = mp.Queue()
                self.analysis_workers.append((SLICAnalysisWorker(
                    self.task_params, data_queue, condition_idx=idx, stim_sites=stim_sites, 
                    title=title), data_queue))

        # Is this a BMI task?
        if 'decoder' in self.task_params:
            data_queue = mp.Queue()
            self.analysis_workers.append((BMIAnalysisWorker(self.task_params, data_queue), data_queue))

        # Start all the workers
        for worker, _ in self.analysis_workers:
            worker.start()
        
    def update(self):
        '''
        Get the latest data from the online server
        '''
        try:
            key, values = self.result_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            return False
        if key == 'state':
            self.state = values[0]
            if values[0] is None:
                print('Experiment finished')
                self.reset()
                self.is_running = False
                self.is_completed = True
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        elif key == 'init' and self.is_running:
            print('Experiment died in error, restarting')
            self.reset()
            self.init()
        elif key == 'init' and values[0]:
            self.is_running = True
            self.init()
        elif key == 'param':
            name, value = values
            self.task_params[name] = value
        else:
            # Send everything else back onto the queues for the analysis workers
            for _, data_queue in self.analysis_workers:
                data_queue.put((key, values))
        return True

    def run(self):
        '''
        Main loop to run the server
        '''
        while True:
            self.update()
            if self._stop_event.is_set():
                break

        self._stop()
        self.sock.close()
        print('OnlineDataServer shut down.')

    def stop(self):
        self._stop_event.set()


if __name__ == '__main__':
    import signal
    import sys

    # Parse arguments
    # online_analysis [hostname] [port] [display]
    if len(sys.argv) >= 2:
        hostname = sys.argv[1]
    else:
        hostname = 'localhost'
    
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    else:
        port = 5000

    if len(sys.argv) >= 4:
        display = sys.argv[3]
    else:
        display = ':0'

    # # Spin up servernode
    # if hostname == '0.0.0.0':
    #     import subprocess
    #     subprocess.Popen('/home/aolab/code/bmi3d/riglib/ecube/servernode-control')

    # Start server
    print(hostname, port, display)
    os.environ['DISPLAY'] = display
    analysis = OnlineDataServer(hostname, port)
    analysis.start()

    # Catch interrupts
    def signal_handler(*args, **kwargs):
        analysis.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

