from riglib import experiment
from features.hdf_features import SaveHDF
from riglib.stereo_opengl.window import FakeWindow
import numpy as np
import matplotlib.pyplot as plt
from tasks import factor_analysis_tasks, bmimultitasks, manualcontrolmultitasks
from features.simulation_features import SimKalmanEnc, SimKFDecoderSup, SimTime
import os, pickle, datetime
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
import shutil, time, tables
from riglib.bmi import feedback_controllers

class SimObsCLDA(SimKalmanEnc, SimTime, factor_analysis_tasks.CLDA_BMIResettingObstacles, FakeWindow):
    def __init__(self, *args, **kwargs):
        dec_fname = os.path.expandvars('$FA_GROM_DATA/grom20160201_01_RMLC02011515.pkl')
        self.decoder = pickle.load(open(dec_fname))
        self.ssm = StateSpaceEndptVel2D()

        A, B, _ = self.ssm.get_ssm_matrices()
        Q = np.mat(np.diag([.5, .5, .5, .1, .1, .1, 0]))
        R = 10**6*np.mat(np.eye(B.shape[1]))
        
        self.fb_ctrl = feedback_controllers.LQRController(A, B, Q, R)
        super(SimObsCLDA, self).__init__(*args, **kwargs)
    def _test_start_trial(self, ts):
        return True

class SimObs(SimKFDecoderSup, SimKalmanEnc, SimTime, bmimultitasks.BMIResettingObstacles, FakeWindow):
    def __init__(self, *args, **kwargs):
        self.ssm = StateSpaceEndptVel2D()

        A, B, _ = self.ssm.get_ssm_matrices()
        Q = np.mat(np.diag([.5, .5, .5, .1, .1, .1, 0]))
        R = 10**6*np.mat(np.eye(B.shape[1]))
        
        self.fb_ctrl = feedback_controllers.LQRController(A, B, Q, R)
        super(SimObs, self).__init__(*args, **kwargs)
    
    def _test_start_trial(self, ts):
        return True


def run_task(session_length):
    Task = experiment.make(SimObsCLDA, [SaveHDF])
    Task.pre_init()

    targets = bmimultitasks.BMIResettingObstacles.centerout_2D_discrete_w_obstacle()

    kwargs=dict(assist_level_time=session_length, assist_level=(0.2, 0.),session_length=session_length,
        half_life=(300., 300.), half_life_time = session_length, timeout_time=60.)

    task = Task(targets, plant_type='cursor_14x14', **kwargs)
    task.init()
    task.run()

    task.decoder.save()

    ct = datetime.datetime.now()
    pnm = os.path.expandvars('$BMI3D/tests/sim_clda/'+ct.strftime("%m%d%y_%H%M") + '.hdf')

    f = open(task.h5file.name)
    f.close()
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, pnm)
    f = open(pnm)
    f.close()
    return pnm
