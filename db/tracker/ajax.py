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

display = Track()

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
    Generic HTTPResponse to wrap JSON data
    '''
    return HttpResponse(json.dumps(data, cls=encoder), mimetype="application/json")

def task_info(request, idx):
    task = Task.objects.get(pk=idx)
    feats = [Feature.objects.get(name=name) for name, isset in request.GET.items() if isset == "true"]
    task_info = dict(params=task.params(feats=feats))

    if issubclass(task.get(feats=feats), experiment.Sequence):
        task_info['sequence'] = task.sequences()

    return _respond(task_info)

def exp_info(request, idx):
    print idx
    entry = TaskEntry.objects.get(pk=idx)
    return _respond(entry.to_json())

def gen_info(request, idx):
    gen = Generator.objects.get(pk=idx)
    return _respond(gen.to_json())

def start_experiment(request, save=True):
    '''
    Handles presses of the 'Start Experiment' and 'Test' buttons in the web 
    interface
    '''
    #make sure we don't have an already-running experiment
    if display.status.value != '':
        return _respond(dict(status="error", msg="Alreading running task!"))

    try:
        data = json.loads(request.POST['data'])
        task =  Task.objects.get(pk=data['task'])
        Exp = task.get(feats=data['feats'].keys())
        entry = TaskEntry(subject_id=data['subject'], task=task)
        params = Parameters.from_html(data['params'])
        entry.params = params.to_json()
        kwargs = dict(subj=entry.subject, task=task, feats=Feature.getall(data['feats'].keys()),
                      params=params.to_json())

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

        # Save the task entry to database
        if save:
            entry.save()
            for feat in data['feats'].keys():
                f = Feature.objects.get(pk=feat)
                entry.feats.add(f.pk)
            response['date'] = entry.date.strftime("%h %d, %Y %I:%M %p")
            response['status'] = "running"
            response['idx'] = entry.id
            kwargs['saveid'] = entry.id
        
        # Start the task FSM and display
        display.runtask(**kwargs)
        logging.info('ajax.start_experiment: Started task with kwargs: %s\n' % str(kwargs))

        # Return the JSON response
        return _respond(response)

    except Exception as e:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))


def stop_experiment(request):
    #make sure that there exists an experiment to stop
    if display.status.value not in ["running", "testing"]:
        return _respond(dict(status="error", msg="No task to end!"))
    try:
        status = display.status.value
        display.stoptask()
        return _respond(dict(status="pending", msg=status))
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))

def enable_clda(request):
    #make sure that there exists an experiment to stop
    if display.status.value not in ["running", "testing"]:
        return _respond(dict(status="error", msg="No task running!"))
    try:
        status = display.status.value
        display.task.enable_clda()
        return _respond(dict(status="pending", msg=status))
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))

def disable_clda(request):
    #make sure that there exists an experiment to stop
    if display.status.value not in ["running", "testing"]:
        return _respond(dict(status="error", msg="No task running!"))
    try:
        status = display.status.value
        display.task.disable_clda()
        return _respond(dict(status="pending", msg=status))
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))        

def save_notes(request, idx):
    te = TaskEntry.objects.get(pk=idx)
    te.notes = request.POST['notes']
    te.save()
    return _respond(dict(status="success"))

def make_bmi(request, idx):
    collide = Decoder.objects.filter(entry=idx, name=request.POST['bminame'])
    if len(collide) > 0:
        return _respond(dict(status='error', msg='Name collision -- please choose a different name'))

    kwargs = dict(
        entry=idx,
        name=request.POST['bminame'],
        clsname=request.POST['bmiclass'],
        extractorname=request.POST['bmiextractor'],
        cells=request.POST['cells'],
        channels=request.POST['channels'],
        binlen=float(request.POST['binlen']),
        tslice=map(float, request.POST.getlist('tslice[]')),
    )
    trainbmi.cache_and_train(**kwargs)
    return _respond(dict(status="success"))
