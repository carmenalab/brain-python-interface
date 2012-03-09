import json
import cPickle
import xmlrpclib

import numpy as np
from django.http import HttpResponse

from models import TaskEntry, Feature, Sequence, Task, Generator, Subject

from riglib import experiment
from tasks import tasklist

from json_param import Parameters
from tasktrack import tracker as display

class encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Parameters):
            return o.params

        return super(encoder, self).default(o)

def _respond(data):
    return HttpResponse(json.dumps(data, cls=encoder), mimetype="application/json")

def task_params(request, taskname):
    feats = [k for k,v in request.GET.items() if v]
    Exp = experiment.make(tasklist[taskname], feats=feats)
    traits = Exp.class_traits()
    data = dict()
    for name in Exp.class_editable_traits():
        desc = traits[name].desc
        val = traits[name].trait_type.default_value
        data[name] = desc, val

    return _respond(data)

def exp_info(request, idx):
    entry = TaskEntry.objects.get(pk=idx)
    sfeats = dict([(f.name, f in entry.feats.all()) for f in Feature.objects.all()])
    params = Parameters(entry.params)
    
    Exp = experiment.make(entry.task.get(), feats=[f.name for f in entry.feats.all()])
    traits = Exp.class_traits()
    traitval = dict()
    for name in Exp.class_editable_traits():
        desc = traits[name].desc
        val = traits[name].trait_type.default_value
        if name in params:
            val = params[name]
        traitval[name] = desc, val
    
    data = dict(task=entry.task_id, params=traitval, notes=entry.notes, features=sfeats, seqid=entry.sequence.id)
    return _respond(data)

def task_seq(request, idx):
    seqs = Sequence.objects.filter(task=idx)
    return _respond(dict([(s.id, s.name) for s in seqs]))

def seq_data(request, idx):
    seq = Sequence.objects.get(pk=idx)
    return _respond(dict(
        task=seq.task_id,
        idx=seq.id, 
        genid=seq.generator.id, 
        params=Parameters(seq.params), 
        static=(seq.sequence != ''),
    ))

def _sequence(task, data, save=True):
    if isinstance(data, dict):
        seqdb = Sequence(generator_id=data['generator'], 
            task=task, name=data['name'], 
            params=Parameters.from_html(data['params']).to_json())
            
        if data['static']:
            seqdb.sequence = seqdb.generator.get()(gen(**params))
        
        if save:
            seqdb.save()
    else:
        seqdb = Sequence.objects.get(pk=data)
    
    return seqdb

def start_experiment(request, save=True):
    #make sure we don't have an already-running experiment
    if display.state is not None:
        return _respond("fail")
    
    data = json.loads(request.POST['data'])
    entry = TaskEntry(subject_id=data['subject_id'], task_id=data['task_id'])
    seq = _sequence(entry.task, data['sequence'])
    entry.sequence = seq

    Exp = experiment.make(entry.task.get(), data['feats'])
    params = Parameters.from_html(data['params'])
    params.trait_norm(Exp.class_traits())
    entry.params = params.to_json()
    
    saveid = None
    if save:
        entry.save()
        for feat in data['feats']:
            f = Feature.objects.get(name=feat)
            entry.feats.add(f.pk)
        saveid = entry.pk

    display.start(entry.task, data['feats'], seq, params.params, saveid)
    return _respond(dict(id=entry.id, subj=entry.subject.name, 
        task=entry.task.name))

def stop_experiment(request):
    #make sure that there exists an experiment to stop
    if display.state not in ["running", "testing"]:
        return _respond("fail")
    return _respond(display.stop())

def report(request):
    if display.state not in ["running", "testing"]:
        return _respond("fail")
    return _respond(display.report())