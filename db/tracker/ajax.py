import json

from django.http import HttpResponse

from track import tracker
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject

from riglib import experiment
from tasks import tasklist

def _respond(data):
	return HttpResponse(json.dumps(data), mimetype="application/json")

def task_params(request, taskname):
	feats = [k for k,v in request.GET.items() if v]
	Exp = experiment.make(tasklist[taskname], feats=feats)
	traits = Exp.class_traits()
	data = dict([ (name, (traits[name].desc, (traits[name].default))) for name in Exp.class_editable_traits()])
	return _respond(data)

def exp_info(request, idx):
	entry = TaskEntry.objects.get(pk=idx)
	sfeats = dict([(f.name, f in entry.feats.all()) for f in Feature.objects.all()])
	params = json.loads(entry.params)
	
	Exp = experiment.make(tasklist[entry.task.name], feats=[f.name for f in entry.feats.all()])
	traits = Exp.class_traits()
	traitval = dict([(name, #-->
			(traits[name].desc, params[name]	if name in params else traits[name].default) )
			for name in Exp.class_editable_traits() ])
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
		params=json.loads(seq.params), 
		static=(seq.sequence != ''),
	))

def start_experiment(request):
	#make sure we don't have an already-running experiment
	if tracker.running:
		return _respond("fail")
	
	data = json.loads(request.POST['data'])
	te = tracker.new(data)
	print te
	return _respond(dict(
		id=te.pk, 
		date=te.date.strftime("%h. %e, %Y, %l:%M %p"), 
		subj=te.subject.name, 
		task=te.task.name))

def stop_experiment(request):
	#make sure that there exists an experiment to stop
	if not tracker.running:
		return _respond("fail")
	tracker.stop()
	return _respond("success")

def report(request):
	#if not tracker.running:
	#	return _respond("fail")
	return _respond(tracker.report())