import json

from django.http import HttpResponse

import views
from models import TaskEntry, Feature, Sequence, Task

from riglib import experiment
from tasks import tasklist

def _respond(data):
	return HttpResponse(json.dumps(data), mimetype="application/json")

def task_params(request, taskname):
	base = tasklist[taskname]
	Exp = experiment.make(base, feats=[k for k,v in request.GET.items() if v])
	traits = Exp.class_traits()
	data = dict([ (name, (traits[name].desc, str(traits[name].default))) for name in Exp.class_editable_traits()])
	return _respond(data)

def exp_info(request, idx):
	entry = TaskEntry.objects.get(pk=idx)
	sfeats = dict([(f.name, f in entry.feats.all()) for f in Feature.objects.all()])
	params = json.loads(entry.params)
	
	Exp = experiment.make(tasklist[entry.task.name], feats=[f.name for f in entry.feats.all()])
	traits = Exp.class_traits()
	traitval = dict([
		(name, (traits[name].desc, str(params[name])
			if name in params else str(traits[name].default)) )
			for name in Exp.class_editable_traits()  ])
	data = {"params":traitval, "notes":entry.notes, "features":sfeats}
	return _respond(data)

def task_seq(request, taskname):
	task = Task.objects.filter(name=taskname)
	seqs = Sequence.objects.filter(task=task[0].id)
	return _respond(dict([(s.id, s.name) for s in seqs]))

def seq_data(request, idx):
	seq = Sequence.objects.get(pk=idx)
	return _respond(dict(idx=seq.id, genid=seq.generator.id, params=json.loads(seq.params)))