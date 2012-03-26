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

def _default_task_params(task, feats, defaults=None):
    if defaults is None:
        defaults = dict()

    Exp = experiment.make(task, feats=feats)
    traits = Exp.class_traits()
    data = dict()

    for name in Exp.class_editable_traits():
        desc = traits[name].desc
        val = traits[name].trait_type.default_value
        if name in defaults:
            val = defaults[name]
        
        if traits[name].trait_type.klass is not None:
            klass = traits[name].trait_type.klass
            for inst, model in namelist.instance_to_model.items():
                if issubclass(klass, inst):
                    models = model.objects.all().order_by("-date")[:10]
                    val = dict([(m.pk, str(m)) for m in models])
                    if name in defaults:
                        val = model.objects.get(pk=defaults[name])
                        val = {val.pk:repr(val)}

        data[name] = desc, val
    
    return data

def task_params(request, taskname):
    feats = [Feature.objects.get(name=name).get() 
        for name,isset in request.GET.items() if isset]
    task = Task.objects.get(name=taskname).get()

    return _respond(_default_task_params(task, feats))

def exp_info(request, idx):
    entry = TaskEntry.objects.get(pk=idx)
    sfeats = dict([(f.name, f in entry.feats.all()) for f in Feature.objects.all()])
    
    feats=[f.get() for f in entry.feats.all()]
    eparams = Parameters(entry.params).params
    params = _default_task_params(entry.task.get(), feats, defaults=eparams)
    
    data = dict(task=entry.task_id, params=params, notes=entry.notes, features=sfeats, seqid=entry.sequence.id)
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
    seq = _sequence(entry.task, data['sequence'], save=save)
    entry.sequence = seq
    feats = [Feature.objects.get(name=n).get() for n in data['feats']]
    Exp = experiment.make(entry.task.get(), feats)
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

    display.start(entry.subject.name, entry.task, feats, seq, params.params, saveid)
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