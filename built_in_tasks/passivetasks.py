'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import os
import tables
import time
import subprocess
import signal

from riglib.experiment import traits, Experiment
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMILoop, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.window import Window, WindowDispl2D

from built_in_tasks.manualcontrolmultitasks import ScreenTargetCapture
from built_in_tasks.bmimultitasks import BMIControlMulti

from .target_graphics import *

bmi_ssm_options = ['Endpt2D', 'Tentacle', 'Joint2L']

class EndPostureFeedbackController(BMILoop, traits.HasTraits):
    ssm_type_options = bmi_ssm_options
    ssm_type = traits.OptionsList(*bmi_ssm_options, bmi3d_input_options=bmi_ssm_options)

    def load_decoder(self):
        self.ssm = StateSpaceEndptVel2D()
        A, B, W = self.ssm.get_ssm_matrices()
        filt = MachineOnlyFilter(A, W)
        units = []
        self.decoder = Decoder(filt, units, self.ssm, binlen=0.1)
        self.decoder.n_features = 1

    def create_feature_extractor(self):
        self.extractor = DummyExtractor()
        self._add_feature_extractor_dtype()


class TargetCaptureVisualFeedback(EndPostureFeedbackController, BMIControlMulti):
    assist_level = (1, 1)
    is_bmi_seed = True

    def move_effector(self):
        pass

class TargetCaptureVFB2DWindow(TargetCaptureVisualFeedback, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(TargetCaptureVFB2DWindow, self).__init__(*args, **kwargs)
        self.assist_level = (1, 1)

    def _start_wait(self):
        self.wait_time = 0.
        super(TargetCaptureVFB2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    @classmethod
    def get_desc(cls, params, report):
        if isinstance(report, list) and len(report) > 0:
            duration = report[-1][-1] - report[0][-1]
            reward_count = 0
            for item in report:
                if item[0] == "reward":
                    reward_count += 1
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        elif isinstance(report, dict):
            duration = report['runtime'] / 60
            reward_count = report['n_success_trials']
            return "{} rewarded trials in {} min".format(reward_count, int(np.ceil(duration / 60)))
        else:
            return "No trials"

from .target_graphics import target_colors

class TargetCaptureReplay(ScreenTargetCapture):
    '''
    Reads the frame-by-frame cursor and trial-by-trial target positions from a saved
    HDF file to display an exact copy of a previous experiment. 
    Doesn't really work, do not recommend using this.
    '''

    hdf_filepath = traits.String("", desc="Filepath of hdf file to replay")

    exclude_parent_traits = list(set(ScreenTargetCapture.class_traits().keys()) - \
        set(['window_size', 'fullscreen']))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t0 = time.perf_counter()
        with tables.open_file(self.hdf_filepath, 'r') as f:
            task = f.root.task.read()
            state = f.root.task_msgs.read()
            trial = f.root.trials.read()
            params = f.root.task.attrs._f_list("user")
            self.task_meta = {k : getattr(f.root.task.attrs, k) for k in params}
        self.replay_state = state
        self.replay_task = task
        self.replay_trial = trial
        for k, v in self.task_meta.items():
            if k in self.exclude_parent_traits:
                print("setting {} to {}".format(k, v))
                setattr(self, k, v)

        # Have to additionally reset the targets since they are created in super().__init__()
        target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        self.targets = [target1, target2]

    def _test_start_trial(self, time_in_state):
        '''Wait for the state change in the HDF file in case there is autostart enabled'''
        trials = self.replay_state[self.replay_state['msg'] == b'target']
        upcoming_trials = [t['time']-1 for t in trials if self.replay_task[t['time']]['trial'] >= self.calc_trial_num()]
        return (np.array(upcoming_trials) <= self.cycle_count).any()

    def _parse_next_trial(self):
        '''Ignore the generator'''
        self.targs = []
        self.gen_indices = []
        trial_num = self.calc_trial_num()
        for trial in self.replay_trial:
            if trial['trial'] == trial_num:
                self.targs.append(trial['target'])
                self.gen_indices.append(trial['index'])

    def _cycle(self):
        '''Have to fudge the cycle_count a bit in case the fps isn't exactly the same'''
        super()._cycle()
        t1 = time.perf_counter() - self.t0
        self.cycle_count = int(t1*self.fps)

    def move_effector(self):
        current_pt = self.replay_task['cursor'][self.cycle_count]
        self.plant.set_endpoint_pos(current_pt)

    def _test_stop(self, ts):
        return super()._test_stop(ts) or self.cycle_count == len(self.replay_task)


class YouTube(Experiment):

    youtube_url = traits.String("", desc="URL pointing to a YouTube video. Only works for videos that support embedding")

    def start_video(self):
        self.video_process = subprocess.Popen(["bash", "../utils/start-youtube.sh", self.youtube_url])

    def stop_video(self):
        os.kill(self.video_process.pid, signal.SIGINT)
        self.video_process.wait()
        
    def _cycle(self):
        try:
            status = self.video_process.poll()
            if status is not None:
                self.state = None
        except:
            pass
        super()._cycle()
   
    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). Starts the YouTube video and stops it after the FSM has finished running
        '''
        try:
            self.start_video()
            super().run()
        finally:
            print("Stopping video")
            self.stop_video()
            
