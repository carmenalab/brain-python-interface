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

import riglib.bmi
from riglib.stereo_opengl import ik
from riglib.experiment import traits, experiment
from riglib.bmi import clda, assist, extractor, train, goal_calculators, ppfdecoder
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.window import WindowDispl2D, FakeWindow
from riglib.bmi.state_space_models import StateSpaceEndptVel2D

from .bmimultitasks import BMIControlMulti


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
