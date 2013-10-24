#!/usr/bin/python
"""
Simulation of CLDA control task
"""
## Imports
from __future__ import division
import numpy as np
import multiprocessing as mp
from riglib.experiment.features import Autostart, SimHDF
import riglib.bmi
from riglib.bmi import kfdecoder, clda
from tasks import bmimultitasks, generatorfunctions as genfns

reload(kfdecoder)
reload(clda)
reload(riglib.bmi)
reload(riglib.bmi.train)

from riglib.stereo_opengl.window import WindowDispl2D

class SimCLDAControlMultiDispl2D(WindowDispl2D, bmimultitasks.SimCLDAControlMulti, Autostart):
    update_rate = 0.1
    def __init__(self, *args, **kwargs):
        self.target_radius = 1.8
        bmimultitasks.SimCLDAControlMulti.__init__(self, *args, **kwargs)
        self.batch_time = 5
        self.half_life  = 20.0

        self.origin_hold_time = 0.250
        self.terminus_hold_time = 0.250
        self.hdf = SimHDF()
        self.task_data = SimHDF()
        self.start_time = 0.
        self.loop_counter = 0
        self.assist_level = 0

    def create_updater(self):
        clda_input_queue = mp.Queue()
        clda_output_queue = mp.Queue()
        self.updater = clda.KFOrthogonalPlantSmoothbatch(clda_input_queue, clda_output_queue,
            self.batch_time, self.half_life)

    def get_time(self):
        return self.loop_counter * 1./60

    def loop_step(self):
        self.loop_counter += 1

gen = genfns.sim_target_seq_generator_multi(8, 1000)
task = SimCLDAControlMultiDispl2D(gen)
task.init()
task.run()
