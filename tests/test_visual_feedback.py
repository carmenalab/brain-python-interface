'''
Test script to run the visual feedback task from the command line.
After the task is finished running, a database record is created and saved and 
the corresponding HDF file is linked.
'''
from db import dbfunctions, json_param
from db.tracker import models, dbq

from riglib import experiment
from features.generator_features import Autostart
from features.hdf_features import SaveHDF
from riglib.experiment import generate


# Tell linux to use Display 0 (the monitor physically attached to the 
# machine. Otherwise, if you are connected remotely, it will try to run 
# the graphics through SSH, which doesn't work for some reason.
import os
os.environ['DISPLAY'] = ':0'

def consolerun(base_class='', feats=[], exp_params=dict(), gen_fn=None, gen_params=dict()):
    if isinstance(base_class, (str, unicode)):
        # assume that it's the name of a task as stored in the database
        base_class = models.Task.objects.get(name=base_class).get()
    
    for k, feat in enumerate(feats):
        # assume that the feature is input as the name of a feature already known to the database
        if isinstance(feat, (str, unicode)):
            feats[k] = models.Feature.objects.get(name=feat).get()

    # Run the pseudo-metaclass constructor
    Exp = experiment.make(base_class, feats=feats)

    # create the sequence of targets
    if gen_fn is None: gen_fn = Exp.get_default_seq_generator()
    targ_seq = gen_fn(**gen_params)

    # instantiate the experiment FSM
    exp = Exp(targ_seq, **exp_params)

    # run!
    exp.run_sync()

consolerun(base_class='machine_control', feats=['saveHDF', 'autostart'], 
    exp_params=dict(session_length=10, plant_visible=True, plant_type='cursor_14x14', rand_start=(0.,0.), max_tries=1), 
    gen_params=dict(nblocks=1)
)

# task = models.Task.objects.get(name='machine_control')
# base_class = task.get()

# feats = [SaveHDF, Autostart]
# Exp = experiment.make(base_class, feats=feats)

# params = 

# # create the sequence of targets
# targ_seq_params = dict(nblocks=1, ntargets=8, boundaries=(-18,18,-12,12), distance=10)
# targ_seq = base_class.centerout_2D_discrete(**targ_seq_params)

# exp = Exp(targ_seq, **params)

# ## Run the task
# exp.run_sync()

save = 0
if save:
    # Save sequence to database
    gen_rec = models.Generator.objects.get(name='centerout_2D_discrete')
    seq_rec = models.Sequence(task=task, generator=gen_rec, name='center_out_testing', params=json_param.Parameters.from_dict(targ_seq_params).to_json())

    seq_rec.save()

    params_obj = json_param.Parameters.from_dict(params)
    
    te = models.TaskEntry()
    subj = models.Subject.objects.get(name='Testing')
    te.subject = subj
    te.task = models.Task.objects.get(name='machine_control')
    te.params = params_obj.to_json()

    te.sequence_id = seq_rec.id
    te.save()
    
    exp.cleanup(dbq, te.id)
