'''
Test experiment finite state machines, task data generation and pygame graphics from the
command line, without frontend overhead. 
'''
from riglib import experiment
from features.generator_features import Autostart
from features.hdf_features import SaveHDF
from riglib.stereo_opengl.window import WindowDispl2D

from riglib.experiment import generate

from built_in_tasks.passivetasks import TargetCaptureVFB2DWindow




def consolerun(base_class='', feats=[], exp_params=dict(), gen_fn=None, gen_params=dict()):
    if isinstance(base_class, str):
        # assume that it's the name of a task as stored in the database
        base_class = models.Task.objects.get(name=base_class).get()
    
    for k, feat in enumerate(feats):
        # assume that the feature is input as the name of a feature already known to the database
        if isinstance(feat, str):
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

if __name__ == '__main__':
    # Tell linux to use Display 0 (the monitor physically attached to the 
    # machine. Otherwise, if you are connected remotely, it will try to run 
    # the graphics through SSH, which doesn't work for some reason.
    import os
    os.environ['DISPLAY'] = ':0'
    
    consolerun(base_class=TargetCaptureVFB2DWindow, feats=[SaveHDF, Autostart], 
        exp_params=dict(session_length=10, window_size=(480, 270), plant_visible=True, plant_type='cursor_14x14', rand_start=(0.,0.), max_tries=1), 
        gen_params=dict(nblocks=1)
    )
