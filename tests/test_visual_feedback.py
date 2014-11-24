'''
Test script to run the visual feedback task from the command line.
After the task is finished running, a database record is created and saved and 
the corresponding HDF file is linked.
'''
from db import dbfunctions
from db.tracker import models

from riglib import experiment
from riglib.stereo_opengl.window import MatplotlibWindow
from features.generator_features import Autostart
from features.hdf_features import SaveHDF

from tasks import generatorfunctions as genfns
from analysis import performance

# Tell linux to use Display 0 (the monitor physically attached to the 
# machine. Otherwise, if you are connected remotely, it will try to run 
# the graphics through SSH, which doesn't work for some reason.
import os
os.environ['DISPLAY'] = ':0'

save = False

task = models.Task.objects.get(name='visual_feedback_multi')
base_class = task.get()

import matplotlib.pyplot as plt

class Debugging(object):
    def __init__(self, *args, **kwargs):
        super(Debugging, self).__init__(*args, **kwargs)

    def init(self):
        super(Debugging, self).init()

    def _start_reward(self, *args, **kwargs):
        print "starting reward!"
        super(Debugging, self)._start_reward(*args, **kwargs)
    
    def _cycle(self):
        #print self.arm.get_endpoint_pos()
        super(Debugging, self)._cycle()


feats = [SaveHDF, Autostart]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=0, plant_visible=True, plant_type='cursor_14x14', 
        rand_start=(0.,0.), max_tries=1)

gen = genfns.sim_target_seq_generator_multi(8, 1000)
exp = Exp(gen, **params)

exp.init()
exp.run()

if save:
    from db import dbfunctions
    from db.tracker import models
    from db.tracker import dbq
    from json_param import Parameters
    params_obj = Parameters.from_dict(params)
    
    te = models.TaskEntry()
    subj = models.Subject.objects.get(name='Testing')
    te.subject = subj
    te.task = models.Task.objects.get(name='visual_feedback_multi')
    te.params = params_obj.to_json()
    te.sequence_id = 0
    te.save()
    
    exp.cleanup(dbq, te.id)
