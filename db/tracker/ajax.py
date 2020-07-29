'''
Handlers for AJAX (Javascript) functions used in the web interface to start 
experiments and train BMI decoders
'''
import json, datetime

import numpy as np
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from riglib import experiment

from .json_param import Parameters

from .models import TaskEntry, Feature, Sequence, Task, Generator, Subject, DataFile, System, Decoder

import trainbmi
import logging
import io, traceback

from . import exp_tracker

http_request_queue = []


def train_decoder_ajax_handler(request, idx):
    '''
    AJAX handler for creating a new decoder.

    Parameters
    ----------
    request : Django HttpRequest
        POST data containing details for how to train the decoder (type, units, update rate, etc.)
    idx : int
        ID number of the models.TaskEntry record with the data used to train the Decoder.

    Returns
    -------
    Django HttpResponse
        Indicates 'success' if all commands initiated without error.
    '''
    ## Check if the name of the decoder is already taken
    collide = Decoder.objects.filter(entry=idx, name=request.POST['bminame'])
    if len(collide) > 0:
        return _respond(dict(status='error', msg='Name collision -- please choose a different name'))
    update_rate = float(request.POST['bmiupdaterate'])

    kwargs = dict(
        entry=idx,
        name=request.POST['bminame'],
        clsname=request.POST['bmiclass'],
        extractorname=request.POST['bmiextractor'],
        cells=request.POST['cells'],
        channels=request.POST['channels'],
        binlen=1./update_rate,
        tslice=list(map(float, request.POST.getlist('tslice[]'))),
        ssm=request.POST['ssm'],
        pos_key=request.POST['pos_key'],
        kin_extractor=request.POST['kin_extractor'],
        zscore=request.POST['zscore'],
    )
    trainbmi.cache_and_train(**kwargs)
    return _respond(dict(status="success"))


class encoder(json.JSONEncoder):
    '''
    Encoder for JSON data that defines how the data should be returned. 
    '''
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Parameters):
            return o.params
        else:
            return super(encoder, self).default(o)

def _respond(data):
    '''
    Generic HTTPResponse to return JSON-formatted dictionary values

    Parameters
    ----------
    data : dict
        Keys and values can be just about anything

    Returns
    -------
    HttpResponse
        JSON-encoded version of the input dictionary
    '''
    return HttpResponse(json.dumps(data, cls=encoder), content_type="application/json")

def task_info(request, idx, dbname='default'):
    '''
    Get information about the task

    Parameters
    ----------
    request : Django HttpRequest

    idx : int
        Primary key used to look up the task from the database

    Returns
    -------
    JSON-encoded dictionary
    '''
    task = Task.objects.using(dbname).get(pk=idx)
    feats = []
    for name, isset in list(request.GET.items()):
        if isset == "true": # box for the feature checked
            feat = Feature.objects.using(dbname).get(name=name)
            feats.append(feat)
    
    task_info = dict(params=task.params(feats=feats), generators=task.get_generators())

    task_cls = task.get(feats=feats)
    if issubclass(task_cls, experiment.Sequence):
        task_info['sequence'] = task.sequences()

    if hasattr(task_cls, 'annotations'):
        task_info['annotations'] = task_cls.annotations
    else:
        task_info['annotations'] = []

    return _respond(task_info)

def exp_info(request, idx, dbname='default'):
    '''
    Get information about the task

    Parameters
    ----------
    request : Django HttpRequest
        POST request triggered by clicking on a task entry from the left side pane
    idx : int
        Primary key used to look up the TaskEntry from the database

    Returns
    -------
    JSON-encoded dictionary 
        Data containing features, parameters, and any report data from the TaskEntry
    '''
    entry = TaskEntry.objects.using(dbname).get(pk=idx)
    try:
        entry_data = entry.to_json()
        tracker = exp_tracker.get()
        entry_data["state"] = tracker.get_status()
    except:
        print("##### Error trying to access task entry data: id=%s, dbname=%s" % (idx, dbname))
        import traceback
        exception = traceback.format_exc()
        exception.replace('\n', '\n    ')
        print(exception.rstrip())
        print("#####")
    else:
        return _respond(entry_data)

def hide_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    print("hide_entry")
    entry = TaskEntry.objects.get(pk=idx)
    entry.visible = False
    entry.save()
    return _respond(dict())

def show_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    print("hide_entry")
    entry = TaskEntry.objects.get(pk=idx)
    entry.visible = True
    entry.save()
    return _respond(dict())

def backup_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    entry = TaskEntry.objects.get(pk=idx)
    entry.backup = True
    entry.save()    
    return _respond(dict())

def unbackup_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    entry = TaskEntry.objects.get(pk=idx)
    entry.backup = False
    entry.save()    
    return _respond(dict())

def gen_info(request, idx):
    try:
        gen = Generator.objects.get(pk=idx)
        return _respond(gen.to_json())
    except:
        traceback.print_exc()

def start_next_exp(request):
    try:
        req, save = http_request_queue.pop(0)
        return start_experiment(req, save=save)
    except IndexError:
        return _respond(dict(status="error", msg="No experiments in queue!"))

@csrf_exempt
def start_experiment(request, save=True, execute=True):
    '''
    Handles presses of the 'Start Experiment' and 'Test' buttons in the browser 
    interface
    '''
    #make sure we don't have an already-running experiment
    tracker = exp_tracker.get()
    if len(tracker.status.value) != 0:
        print("Task is running, exp_tracker.status.value:", tracker.status.value)
        return _respond(dict(status="running", msg="Already running task!"))

    # Try to start the task, and if there are any errors, send them to the browser interface
    try:        
        data = json.loads(request.POST['data'])

        task =  Task.objects.get(pk=data['task'])
        feature_names = list(data['feats'].keys())
        Exp = task.get(feats=feature_names)

        entry = TaskEntry.objects.create(subject_id=data['subject'], task_id=task.id)
        if 'entry_name' in data:
            entry.entry_name = data['entry_name']
        if 'date' in data and data['date'] != "Today" and len(data['date'].split("-")) == 3:
            datestr = data['date'].split("-")
            print("Got custom date: ", datestr)
            entry.date = datetime.datetime(int(datestr[0]), int(datestr[1]), int(datestr[2])) # this does not work: datetime.datetime.strptime("%Y-%m-%d", datetime.datetime.now().strftime("%Y-%m-%d"))

        params = Parameters.from_html(data['params'])
        entry.params = params.to_json()
        feats = Feature.getall(feature_names)
        kwargs = dict(subj=entry.subject.id, base_class=task.get(), 
            feats=feats, params=params)

        # Save the target sequence to the database and link to the task entry, if the task type uses target sequences
        if issubclass(Exp, experiment.Sequence):
            print("creating seq")
            print("data['sequence'] POST data")
            print(data['sequence'])
            seq = Sequence.from_json(data['sequence'])
            seq.task = task
            if save:
                seq.save()
            entry.sequence = seq
            kwargs['seq'] = seq
        
        response = dict(status="testing", subj=entry.subject.name, 
                        task=entry.task.name)

        if save:
            # tag software version using the git hash
            import git
            repo = git.repo.Repo(__file__, search_parent_directories=True)
            sw_version = repo.commit().hexsha[:8]
            repo_dirty = repo.is_dirty(index=True, working_tree=True, untracked_files=False)
            if repo_dirty:
                sw_version += '.dirty'
            entry.sw_version = sw_version
            
            # Save the task entry to database
            entry.save()

            # Link the features used to the task entry
            for feat_name in feature_names:
                f = Feature.objects.get(name=feat_name)
                entry.feats.add(f.pk)

            response['date'] = entry.date.strftime("%h %d, %Y %I:%M %p")
            response['status'] = "running"
            response['idx'] = entry.id

            # Give the entry ID to the runtask as a kwarg so that files can be linked after the task is done
            kwargs['saveid'] = entry.id
        else:
            entry.delete()
        
        # Start the task FSM and tracker
        if execute:
            tracker.runtask(**kwargs)
        else:
            response["status"] = "completed"

        # Return the JSON response
        return _respond(response)

    except Exception as e:
        # Generate an HTML response with the traceback of any exceptions thrown
        import io
        import traceback
        err = io.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        traceback.print_exc()
        return _respond(dict(status="error", msg=err.read()))

def rpc(fn):
    '''
    Generic remote procedure call function

    Parameters
    ----------
    fn : callable
        Function which takes a single argument, the tracker object. 
        Return values from this function are ignored.

    Returns
    -------
    JSON-encoded dictionary 
    '''
    tracker = exp_tracker.get()

    # make sure that there exists an experiment to interact with
    if tracker.status.value not in [b"running", b"testing"]:
        print("rpc not possible", str(tracker.status.value))
        return _respond(dict(status="error", msg="No task running, so cannot run command!"))

    try:
        status = tracker.status.value.decode("utf-8")
        fn_response = fn(tracker)
        response_data = dict(status="pending", msg=status)
        if not fn_response is None:
            response_data['data'] = fn_response
        return _respond(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _respond_err(e)

def _respond_err(e):
    '''
    Default error response from server to webclient

    Parameters
    ----------
    e : Exception
        Error & traceback to convert to string format. 

    Returns
    -------
    JSON-encoded dictionary
        Sets status to "error" and provides the specific error message
    '''
    err = io.StringIO()
    traceback.print_exc(None, err)
    err.seek(0)
    return _respond(dict(status="error", msg=err.read()))        

@csrf_exempt
def stop_experiment(request):
    return rpc(lambda tracker: tracker.stoptask())

def enable_clda(request):
    return rpc(lambda tracker: tracker.task_proxy.enable_clda())

def disable_clda(request):
    return rpc(lambda tracker: tracker.task_proxy.disable_clda())

def set_task_attr(request, attr, value):
    '''
    Generic function to change a task attribute while the task is running.
    '''
    return rpc(lambda tracker: tracker.task_proxy.remote_set_attr(attr, value))

@csrf_exempt
def save_notes(request, idx):
    te = TaskEntry.objects.get(pk=idx)
    te.notes = request.POST['notes']
    te.save()
    return _respond(dict(status="success"))

def reward_drain(request, onoff):
    '''
    Start/stop the "drain" of a solenoid reward remotely
    '''
    from riglib import reward
    r = reward.Basic()

    if onoff == 'on':
        r.drain(600)
        print('drain on')
    else:
        print('drain off')
        r.drain_off()
    return HttpResponse('Turning reward %s' % onoff)

def populate_models(request):
    """ Database initialization code. When 'db.tracker' is imported, it goes through the database and ensures that 
    1) at least one subject is present
    2) all the tasks from 'tasklist' appear in the db
    3) all the features from 'featurelist' appear in the db
    4) all the generators from all the tasks appear in the db 
    """
    from . import models
    subjects = models.Subject.objects.all()
    if len(subjects) == 0:
        subj = models.Subject(name='testing')
        subj.save()

    for m in [models.Task, models.Feature, models.Generator, models.System]:
        m.populate()

    return HttpResponse("Updated Tasks, features generators, and systems")

@csrf_exempt
def add_new_task(request):
    from . import models
    name, import_path = request.POST['name'], request.POST['import_path']
    
    #  verify import path
    if import_path == '':
        import_path = "riglib.experiment.Experiment"

    try:
        models.import_by_path(import_path)
    except:
        import traceback
        traceback.print_exc()
        return _respond(dict(msg="import path invalid!", status="error"))

    task = models.Task(name=name, import_path=import_path)
    task.save()

    # add any new generators for the task
    models.Generator.populate()

    task_data = dict(id=task.id, name=task.name, import_path=task.import_path)
    return _respond(dict(msg="Added new task: %s" % task.name, status="success", data=task_data))

@csrf_exempt
def update_task_import_path(request):
    from . import models
    task_id, import_path = request.POST['id'], request.POST['import_path']

    # verify import path
    if import_path == '':
        import_path = "riglib.experiment.Experiment"

    try:
        models.import_by_path(import_path)
    except:
        import traceback
        traceback.print_exc()
        return _respond(dict(msg="import path invalid!", status="error"))

    task = models.Task.objects.get(id=task_id)
    task.import_path = import_path
    task.save()

    task_data = dict(id=task.id, name=task.name, import_path=task.import_path)
    return _respond(dict(msg="Updated task import path: %s" % task.name, status="success", data=task_data))

@csrf_exempt
def add_new_subject(request):
    from . import models 
    subject_name = request.POST['subject_name']
    subj = models.Subject(name=subject_name)
    subj.save()

    return _respond(dict(msg="Added new subject: %s" % subj.name, status="success", data=dict(id=subj.id, name=subj.name)))

@csrf_exempt
def add_new_system(request):
    from . import models
    sys = models.System(name=request.POST['name'], path=request.POST['path'], 
        processor_path=request.POST['processor_path'])
    sys.save()

    return _respond(dict(msg="Added new system: %s" % sys.name, status="success"))

@csrf_exempt
def enable_features(request):
    from features import built_in_features
    from . import models

    feature_names_added = []

    for key in request.POST:
        if key in built_in_features:
            # check if the feature is already installed
            existing_features = models.Feature.objects.filter(name=key)

            if len(existing_features) > 0:
                continue

            import_path = built_in_features[key].__module__ + '.' + built_in_features[key].__qualname__
            feat = models.Feature(name=key, import_path=import_path)
            feat.save()

            feature_names_added.append(feat.name)

    return _respond(dict(msg="Enabled built-in features: %s" % str(feature_names_added), status="success"))

@csrf_exempt
def add_new_feature(request):
    from . import models
    name, import_path = request.POST['name'], request.POST['import_path']
    feat = models.Feature(name=name, import_path=import_path)
    feat.save()

    return _respond(dict(msg="Added new feature: %s" % feat.name, status="success"))

@csrf_exempt
def update_feature_import_path(request):
    feature_id, feature_path = request.POST['id'], request.POST['import_path']

    try:
        models.import_by_path(import_path)
    except:
        import traceback
        traceback.print_exc()
        return _respond(dict(msg="import path invalid!", status="error"))    

    feat = models.Feature.objects.get(id=feature_id)
    feat.import_path = import_path
    feat.save()

    return _respond(dict(msg="Updated feature import path: %s" % feat.name, status="success"))

@csrf_exempt
def setup_run_upkeep(request):
    # Update the list of generators
    from . import models
    models.Generator.populate()
    return HttpResponse("Updated generators!")

@csrf_exempt
def get_report(request):
    '''
    Get data for the report field in the frontend
    '''
    def report_fn(tracker):
        tracker.task_proxy.update_report_stats()
        reportstats = tracker.task_proxy.reportstats
        return reportstats

    return rpc(report_fn)

@csrf_exempt
def record_annotation(request):
    return rpc(lambda tracker: tracker.task_proxy.record_annotation(request.POST["annotation"]))
    
@csrf_exempt
def get_status(request):
    """ Send the task tracker's status back to the frontend """
    tracker = exp_tracker.get()
    if tracker.task_kwargs is None:
        saveid = None
    else:
        saveid = tracker.task_kwargs["saveid"]
    print("saveid", saveid)
    return _respond(dict(status=tracker.get_status(), saveid=saveid))

@csrf_exempt
def save_entry_name(request):
    from . import models
    te_rec = models.TaskEntry.objects.get(id=request.POST["id"])
    te_rec.entry_name = request.POST["entry_name"]
    te_rec.save()
    return _respond(dict(status="success", msg="Saved entry name: %s" % te_rec.entry_name))

@csrf_exempt
def update_built_in_feature_import_paths(request):
    """For built-in features, update the import path based on the features module"""
    from . import models
    for feat in models.Feature.objects.all():
        feat.get(update_builtin=True)
    return _respond(dict(status="success", msg="Updated built-in feature paths!"))

def update_database_storage_path(request):
    from . import models
    db_name = request.POST['db_name']
    db_storage_path = request.POST['db_storage_path']

    if db_name == 'default':
        models.KeyValueStore.set("data_path", db_storage_path)
        return _respond(dict(status="success", msg="Updated storage path for %s db" % db_name))
    else:
        return _respond(dict(status="error", msg="Not yet implemented for non-default tables!"))

def save_recording_sys(request):
    from . import models
    models.KeyValueStore.set('recording_sys', request.POST['selected_recording_sys'])
    print(models.KeyValueStore.get('recording_sys'))
    ret_msg = "Set recording_sys to %s" % models.KeyValueStore.get('recording_sys')
    return _respond(dict(status="success", msg=ret_msg))

@csrf_exempt
def setup_handler(request):
    """One-stop handler for setup functions to avoid adding a bunch of URLs"""
    action = request.POST['action']
    if action == "update_database_storage_path":
        return update_database_storage_path(request)
    elif action == "save_recording_sys":
        return save_recording_sys(request)
    else:
        return _respond(dict(status="error", msg="Unrecognized data type: %s" % data_type))

