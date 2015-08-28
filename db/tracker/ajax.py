'''
Handlers for AJAX (Javascript) functions used in the web interface to start 
experiments and train BMI decoders
'''
import json

import numpy as np
from django.http import HttpResponse

from riglib import experiment

from json_param import Parameters
from tasktrack import Track
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject, DataFile, System, Decoder

import trainbmi
import logging

exp_tracker = Track()

http_request_queue = []

# create the object representing the reward system. 
try:
    from riglib import reward
    r = reward.Basic()
except:
    pass

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
        tslice=map(float, request.POST.getlist('tslice[]')),
        ssm=request.POST['ssm'],
        pos_key=request.POST['pos_key'],
        kin_extractor=request.POST['kin_extractor'],
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
    return HttpResponse(json.dumps(data, cls=encoder), mimetype="application/json")

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
    feats = [Feature.objects.using(dbname).get(name=name) for name, isset in request.GET.items() if isset == "true"]
    task_info = dict(params=task.params(feats=feats))

    if issubclass(task.get(feats=feats), experiment.Sequence):
        task_info['sequence'] = task.sequences()

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
    print idx, dbname
    entry = TaskEntry.objects.using(dbname).get(pk=idx)
    return _respond(entry.to_json())

def hide_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    print "hide_entry"
    entry = TaskEntry.objects.get(pk=idx)
    entry.visible = False
    entry.backup = False
    entry.save()
    return _respond(dict())

def backup_entry(request, idx):
    '''
    See documentation for exp_info
    '''
    entry = TaskEntry.objects.get(pk=idx)
    entry.visible = True
    entry.backup = True
    entry.save()    
    return _respond(dict())

def gen_info(request, idx):
    gen = Generator.objects.get(pk=idx)
    return _respond(gen.to_json())

def start_next_exp(request):
    try:
        req, save = http_request_queue.pop(0)
        return start_experiment(req, save=save)
    except IndexError:
        return _respond(dict(status="error", msg="No experiments in queue!"))

def start_experiment(request, save=True):
    '''
    Handles presses of the 'Start Experiment' and 'Test' buttons in the browser 
    interface
    '''
    #make sure we don't have an already-running experiment
    if exp_tracker.status.value != '':
        http_request_queue.append((request, save))
        return _respond(dict(status="running", msg="Already running task, queuelen=%d!" % len(http_request_queue)))

    # Try to start the task, and if there are any errors, send them to the browser interface
    try:
        data = json.loads(request.POST['data'])

        task =  Task.objects.get(pk=data['task'])
        Exp = task.get(feats=data['feats'].keys())

        entry = TaskEntry(subject_id=data['subject'], task=task)
        params = Parameters.from_html(data['params'])
        entry.params = params.to_json()
        kwargs = dict(subj=entry.subject, task=task, feats=Feature.getall(data['feats'].keys()),
                      params=params)

        # Save the target sequence to the database and link to the task entry, if the task type uses target sequences
        if issubclass(Exp, experiment.Sequence):
            seq = Sequence.from_json(data['sequence'])
            seq.task = task
            if save:
                seq.save()
            entry.sequence = seq
            kwargs['seq'] = seq
        else:
            entry.sequence_id = -1
        
        response = dict(status="testing", subj=entry.subject.name, 
                        task=entry.task.name)

        if save:
            # Save the task entry to database
            entry.save()

            # Link the features used to the task entry
            for feat in data['feats'].keys():
                f = Feature.objects.get(pk=feat)
                entry.feats.add(f.pk)

            response['date'] = entry.date.strftime("%h %d, %Y %I:%M %p")
            response['status'] = "running"
            response['idx'] = entry.id

            # Give the entry ID to the runtask as a kwarg so that files can be linked after the task is done
            kwargs['saveid'] = entry.id
        
        # Start the task FSM and exp_tracker
        exp_tracker.runtask(**kwargs)

        # Return the JSON response
        return _respond(response)

    except Exception as e:
        # Generate an HTML response with the traceback of any exceptions thrown
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))

def rpc(fn):
    '''
    Generic remote procedure call function

    Parameters
    ----------
    fn : callable
        Function which takes a single argument, the exp_tracker object. 
        Return values from this function are ignored.

    Returns
    -------
    JSON-encoded dictionary 
    '''
    #make sure that there exists an experiment to stop
    if exp_tracker.status.value not in ["running", "testing"]:
        return _respond(dict(status="error", msg="No task to modify attributes"))
    try:
        status = exp_tracker.status.value
        fn(exp_tracker)
        return _respond(dict(status="pending", msg=status))
    except Exception as e:
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
    import cStringIO
    import traceback
    err = cStringIO.StringIO()
    traceback.print_exc(None, err)
    err.seek(0)
    return _respond(dict(status="error", msg=err.read()))        

def stop_experiment(request):
    return rpc(lambda exp_tracker: exp_tracker.stoptask())

def enable_clda(request):
    return rpc(lambda exp_tracker: exp_tracker.task.enable_clda())

def disable_clda(request):
    return rpc(lambda exp_tracker: exp_tracker.task.disable_clda())

def set_task_attr(request, attr, value):
    '''
    Generic function to change a task attribute while the task is running.
    '''
    return rpc(lambda exp_tracker: exp_tracker.task.remote_set_attr(attr, value))

def save_notes(request, idx):
    te = TaskEntry.objects.get(pk=idx)
    te.notes = request.POST['notes']
    te.save()
    return _respond(dict(status="success"))

def reward_drain(request, onoff):
    '''
    Start/stop the "drain" of a solenoid reward remotely
    '''
    if onoff == 'on':
        r.drain(600)
        print 'drain on'
    else:
        print 'drain off'
        r.drain_off()
    return HttpResponse('Turning reward %s' % onoff)
