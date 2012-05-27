import json
import cPickle
import xmlrpclib

import numpy as np
from django.http import HttpResponse

from riglib import experiment

import namelist
from json_param import Parameters
from tasktrack import tracker as display
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject

class encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Parameters):
            return o.params

        return super(encoder, self).default(o)

def _respond(data):
    return HttpResponse(json.dumps(data, cls=encoder), mimetype="application/json")

def task_info(request, idx):
    task = Task.objects.get(pk=idx)
    feats = [Feature.objects.get(name=name) for name, isset in request.GET.items() if isset == "true"]
    task_info = dict(params=task.params(feats=feats))

    if issubclass(task.get(feats=feats), experiment.Sequence):
        task_info['sequence'] = task.sequences()

    return _respond(task_info)

def exp_info(request, idx):
    entry = TaskEntry.objects.get(pk=idx)
    return _respond(entry.to_json())

def gen_info(request, idx):
    gen = Generator.objects.get(pk=idx)
    return _respond(gen.to_json())

def start_experiment(request, save=True):
    #make sure we don't have an already-running experiment
    if display.state is not None:
        return _respond(dict(state="fail"))
    try:
        data = json.loads(request.POST['data'])
        task =  Task.objects.get(pk=data['task'])
        Exp = task.get(feats=data['feats'].keys())
        entry = TaskEntry(subject_id=data['subject'], task=task)
        params = Parameters.from_html(data['params'])
        params.trait_norm(Exp.class_traits())
        entry.params = params.to_json()
        kwargs = dict(task=task, feats=Feature.getall(data['feats'].keys()), params=params.params)

        if issubclass(Exp, experiment.Sequence):
            seq = Sequence.from_json(data['sequence'])
            entry.sequence = seq
            if save:
                seq.save()
            kwargs['seq'] = seq
        else:
            entry.sequence_id = -1
        
        saveid = None
        if save:
            kwargs['saveid'] = saveid
            entry.save()
            for feat in data['feats']:
                f = Feature.objects.get(name=feat)
                entry.feats.add(f.pk)
        
        display.start(**kwargs)
        return _respond(dict(state="running", id=entry.id, subj=entry.subject.name, 
            task=entry.task.name))

    except Exception as e:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(state="error", msg=err.read()))


def stop_experiment(request):
    #make sure that there exists an experiment to stop
    if display.state not in ["running", "testing"]:
        return _respond("fail")
    return _respond(display.stop())

def report(request):
    if display.state not in ["running", "testing"]:
        return _respond("fail")
    return _respond(display.report())