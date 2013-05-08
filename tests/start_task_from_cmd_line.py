import os
import json
import cPickle
import xmlrpclib

import numpy as np
from django.http import HttpResponse

from riglib import experiment

import namelist
from json_param import Parameters
from tasktrack import Track
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject, DataFile, System, Decoder

import trainbmi

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

response = dict(status="testing", subj=entry.subject.name, task=entry.task.name)
if save:
    entry.save()
    for feat in data['feats'].keys():
        f = Feature.objects.get(pk=feat)
        entry.feats.add(f.pk)
    response['date'] = entry.date.strftime("%h %d, %Y %I:%M %p")
    response['status'] = "running"
    response['idx'] = entry.id
    kwargs['saveid'] = entry.id

display.runtask(**kwargs)
return _respond(response)

