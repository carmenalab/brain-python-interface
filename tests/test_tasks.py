from built_in_tasks.manualcontrolmultitasks import TrackingTask, rotations, ManualControl, ScreenTargetTracking
from built_in_tasks.othertasks import Conditions, LaserConditions
from built_in_tasks.target_capture_task import ScreenTargetCapture
from built_in_tasks.passivetasks import YouTube
from features.hdf_features import SaveHDF
from riglib.stereo_opengl.window import WindowDispl2D
from riglib import experiment
from features.peripheral_device_features import MouseControl
from features.optitrack_features import OptitrackSimulate, Optitrack
from features.reward_features import ProgressBar
import cProfile
import pstats
from riglib.stereo_opengl.window import Window, Window2D 
import unittest
import numpy as np
import os
import socket

def init_exp(base_class, feats, seq=None, **kwargs):
    hostname = socket.gethostname()
    if hostname in ['pagaiisland2']:
        os.environ['DISPLAY'] = ':0.1'
    Exp = experiment.make(base_class, feats=feats)
    if seq is not None:
        exp = Exp(seq, **kwargs)
    else:
        exp = Exp(**kwargs)
    exp.init()
    return exp

class TestManualControlTasks(unittest.TestCase):

    @unittest.skip("")
    def test_exp(self):
        seq = ManualControl.centerout_2D()
        exp = init_exp(ManualControl, [MouseControl, Window2D], seq)
        exp.rotation = 'xzy'
        exp.run()
    
    @unittest.skip("")
    def test_tracking(self):
        print("Running tracking task test")
        seq = TrackingTask.tracking_target_debug(nblocks=1, ntrials=6, time_length=5, seed=40, sample_rate=60, ramp=1) # sample_rate needs to match fps in ScreenTargetTracking
        exp = init_exp(TrackingTask, [MouseControl, Window2D], seq) # , window_size=(1000,800)
        exp.rotation = 'xzy'
        exp.run()

    # @unittest.skip("only to test progress bar")
    # def test_tracking(self):
    #     seq = TrackingTask.tracking_target_debug(nblocks=1, ntrials=6, time_length=5, seed=40, sample_rate=60, ramp=1) # sample_rate needs to match fps in ScreenTargetTracking
    #     exp = init_exp(TrackingTask, [MouseControl, Window2D, ProgressBar], seq)
    #     exp.rotation = 'xzy'
    #     exp.run()

class TestSeqGenerators(unittest.TestCase):

    # @unittest.skip("")
    def test_gen_ascending(self):
        seq = Conditions.gen_conditions(3, [1, 2], ascend=True)
        self.assertSequenceEqual(seq[0], [0, 0, 0, 1, 1, 1])

    # @unittest.skip("")
    def test_gen_out_2D(self):
        seq = ScreenTargetCapture.out_2D(nblocks=1, )
        seq = list(seq)
        idx = np.array([s[0][0] for s in seq])
        loc = np.array([s[1][0] for s in seq])
        print(idx)
        print(loc)
        self.assertCountEqual(idx, [1, 2, 3, 4, 5, 6, 7, 8])

        # Target 1 should be 12 o'clock
        self.assertAlmostEqual(loc[idx == 1, 0][0], 0)
        self.assertAlmostEqual(loc[idx == 1, 2][0], 10)

        # Target 3 should be 3 o'clock
        self.assertAlmostEqual(loc[idx == 3, 0][0], 10)
        self.assertAlmostEqual(loc[idx == 3, 2][0], 0)

    # @unittest.skip("")
    def test_dual_laser_wave(self):
        seq = LaserConditions.dual_laser_square_wave(duty_cycle_1=0.025, duty_cycle_2=0.025, phase_delay_2=0.1)
        print(seq[0])

    def test_corners(self):
        seq = ScreenTargetCapture.corners_2D(chain_length=3)
        seq = list(seq)

        idx = np.array([s[0][0] for s in seq])
        loc = np.array([s[1][0] for s in seq])
        print("corners---------------")
        print(idx)
        print(loc)
        print("---------------corners")

class TestYouTube(unittest.TestCase):

    @unittest.skip("")
    def test_youtube_exp(self):

        exp = init_exp(YouTube, [], youtube_url="https://www.youtube.com/watch?v=Qe9ansjvF7M")
        exp.run()

if __name__ == '__main__':
    unittest.main()