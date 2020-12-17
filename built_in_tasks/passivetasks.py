'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import time
import os
import pdb
import multiprocessing as mp
import pickle
import tables
import re
import tempfile, traceback, datetime
import pygame

import riglib.bmi
from riglib.stereo_opengl import ik
from riglib.experiment import traits, experiment
from riglib.bmi import clda, assist, extractor, train, goal_calculators, ppfdecoder
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.window import Window, WindowDispl2D, FakeWindow
from riglib.experiment import Sequence

from built_in_tasks.bmimultitasks import BMIControlMulti
from built_in_tasks.target_graphics import VirtualCircularTarget, target_colors

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

class MonkeyTraining(Sequence, Window):
    '''Simplified target capture task'''

    status = dict(
        wait = dict(start_trial="trial", stop=None),
        trial = dict(end_trial="wait"),
    )

    state = "wait"
    sequence_generators = ['static', 'rand_pt_to_pt']
    
    background = (0,0,0,1)
    target_color = traits.OptionsList(tuple(target_colors.keys()), desc="Color of the target")
    target_radius = traits.Float(5, desc="Radius of targets in cm")
    
    wait_time = traits.Float(5.0, desc="Time in between trials (s). If set to 0, then trials begin with keypress")
    trial_time = traits.Float(1.0, desc="Trial length (s)")

    is_bmi_seed = True

    def _test_start_trial(self, ts):
        if self.wait_time == 0:
            from pygame import K_ESCAPE
            return self.event is not None and self.event[0] != K_ESCAPE # keypress
        else:
            return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_location = np.array([0, 0, 0])
        self.target = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        for model in self.target.graphics_models:
            self.add_model(model)

    def _parse_next_trial(self):
        self.target_location = self.next_trial

    def _start_trial(self):
        self.target.move_to_position(self.target_location)
        self.target.show()

    def _start_wait(self):
        self.target.hide()
        Sequence._start_wait(self)

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        super().init()

    def _cycle(self):
        self.task_data['target'] = self.target_location.copy()
        super()._cycle()

    @staticmethod
    def static(pos=(0,0,0), ntrials=0):
        '''Single location, finite (ntrials!=0) or infinite (ntrials==0)'''
        if ntrials == 0:
            while True:
                yield np.array(pos)
        else:
            return np.tile(pos, (ntrials,1))

    @staticmethod
    def rand_pt_to_pt(seq_len=100, boundaries=(-18,18,-12,12), buf=2):
        '''
        Generates sequences of random postiions in the XZ plane

        Parameters
        ----------
        length : int
            The number of targets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        buf : float
            The distance from the boundary to the center of a target.        

        Returns
        -------
        list
            Each element of the list is an array of shape (seq_len, 3) indicating the target 
            positions to be acquired for the trial.
        '''
        xmin, xmax, zmin, zmax = boundaries
        pts = np.vstack([np.random.uniform(xmin+buf, xmax-buf, seq_len),
            np.zeros(seq_len), np.random.uniform(zmin+buf, zmax-buf, seq_len)]).T
        return pts
