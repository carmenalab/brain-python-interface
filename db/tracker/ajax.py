import json
import cPickle

from django.http import HttpResponse

from . import expqueue
import views
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject

from riglib import experiment
from tasks import tasklist

def _respond(data):
	return HttpResponse(json.dumps(data), mimetype="application/json")

def task_params(request, taskname):
	feats = [k for k,v in request.GET.items() if v]
	Exp = experiment.make(tasklist[taskname], feats=feats)
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
	data = dict(params=traitval, notes=entry.notes, features=sfeats, seqid=entry.sequence.id)
	return _respond(data)

def task_seq(request, idx):
	seqs = Sequence.objects.filter(task=idx)
	return _respond(dict([(s.id, s.name) for s in seqs]))

def seq_data(request, idx):
	seq = Sequence.objects.get(pk=idx)
	return _respond(dict(
		idx=seq.id, 
		genid=seq.generator.id, 
		params=json.loads(seq.params), 
		static=(seq.sequence != ''),
	))

def start_experiment(request):
	data = json.loads(request.POST['data'])
	subj = Subject.objects.get(pk=data['subject'])
	task = Task.objects.get(pk=data['task'])

	Exp = experiment.make(tasklist[task.name], feats=data['feats'])
	if not isinstance(data['sequence'], dict):
		#Existing sequence --- load it up
		seqdb = Sequence.objects.get(pk=int(data['sequence']))
		if seqdb.sequence != "":
			gen = experiment.generate.runseq(Exp, cPickle.loads(seqdb.sequence))
		else:
			params = json.loads(seqdb.params)
			gen = experiment.genlist[seqdb.generator.name]
			if seqdb.generator.static:
				del params['length']
				gen = experiment.generate.runseq(Exp, gen(params['length'], **params))
			else:
				gen = gen(Exp, **params)
	else:
		#New sequence! Generate it from scratch and commit it to the database
		gendb = Generator.objects.get(pk=data['sequence']['generator'])
		seqdb = Sequence(generator=gendb, name=data['sequence']['name'], task=task)
		gen = experiment.genlist[gendb.name]
		if gendb.static:
			length = int(data['sequence']['params']['length'])
			del data['sequence']['params']['length']
			params = dict([(k, json.loads(v)) for k, v in data['sequence']['params'].items()])
			seq = gen(length, **params)
			seqdb.params = json.dumps(params)
			if data['sequence']['static']:
				seqdb.sequence = seq

			gen = experiment.generate.runseq(Exp, seq)
		else:
			params = dict([(k, json.loads(v)) for k, v in data['sequence']['params'].items()])
			seqdb.params = json.dumps(params)
			gen = gen(Exp, **params)
		seqdb.save()
	
	traits = Exp.class_traits()
	params = dict()
	for k, v in data['params'].items():
		print v
		v = json.loads(v)
		ttype = traits[k].trait_type.__class__
		if ttype in experiment.typemap:
			v = experiment.typemap[ttype](v)
		else:
			print ttype
		params[k] = v

	#TaskEntry(subject=int(data['subject']), task=int(data['task']), )

	exp = Exp(gen, **params)
	exp.start()
	expqueue.append(exp)

	return HttpResponse("blah")