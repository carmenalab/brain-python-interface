import time
import traceback
import numpy as np
import os
import socket
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
os.environ["BMI3D_PORT"] = "8000"

from db.tracker.models import TaskEntry, Feature, Sequence, Task, Subject, Experimenter
from db.tracker.json_param import Parameters
from db.tracker.tasktrack import Track
from db.boot_django import boot_django
boot_django()

from .. import experiment

def run_experiment(subject_name, experimenter_name, project, session, 
                   task_name, feat_names, seq_name=None, task_desc=None, save=True,
                   **kwargs):
    '''
    Run a task with the given database IDs and parameters. Just like running an experiment in the GUI,
    the task is added to the database. This is a blocking call.
    '''
    tracker = Track.get_instance()
    if tracker.task_proxy is not None:
        print("Task is running, cannot start new task")
        return
    
    hostname = socket.gethostname()
    if hostname in ['pagaiisland2', 'human-bmi']:
        os.environ['DISPLAY'] = ':0.1'

    task =  Task.objects.get(name=task_name)
    subject = Subject.objects.get(name=subject_name)
    experimenter = Experimenter.objects.get(name=experimenter_name)

    entry = TaskEntry.objects.create(rig_name=hostname, subject_id=subject.id, task_id=task.id, experimenter_id=experimenter.id,
        project=project, session=session)
    entry.entry_name = task_desc

    # Link the features used to the task entry
    for feat_name in feat_names:
        f = Feature.objects.get(name=feat_name)
        entry.feats.add(f.pk)

    js_params = entry.task.params(entry.feats.all(), values=kwargs)
    params = {k: js_params[k]['value'] if 'value' in js_params[k] else js_params[k]['default'] for k in js_params.keys()}
    params = Parameters.from_dict(params)
    print('params:', params.to_json())
    entry.params = params.to_json()
    feats = Feature.getall(feat_names)

    kwargs = dict(subj=entry.subject.id, subject_name=subject_name, base_class=task.get(),
            feats=feats, params=params)

    # Save the target sequence to the database and link to the task entry, if the task type uses target sequences
    if issubclass(task.get(feats=feat_names), experiment.Sequence):
        seq = Sequence.objects.get(name=seq_name)
        if seq is None:
            raise ValueError(f"Sequence '{seq_name}' not found in database")
        entry.sequence = seq
        kwargs['seq'] = seq

    # Save the task entry to database
    if save:
        entry.sw_version = 'script'
        entry.save()

        # Give the entry ID to the runtask as a kwarg so that files can be linked after the task is done
        kwargs['saveid'] = entry.id
    else:
        entry.delete()

    # Use the singleton tasktracker object to start the task
    tracker = Track.get_instance()
    tracker.runtask(cli=True, **kwargs)

    try:
        while (tracker.proc is not None) and (tracker.proc.is_alive()):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        tracker.stoptask()
    except:
        print("Error in task:", tracker.get_status())
        traceback.print_exc()
    finally:
        tracker.reset()