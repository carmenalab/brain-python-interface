import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))
import json

from django.shortcuts import render_to_response

from tracker.models import TaskEntry, Task, Subject, Feature

from riglib import experiment
from tasks import tasklist
from riglib.experiment import featlist

def list(request):
	entries = TaskEntry.objects.all().order_by("-date")[:100]
	tlist = Task.objects.all().order_by("name")
	subjects = Subject.objects.all().order_by("name")
	return render_to_response('list.html', dict(entries=entries, subjects=subjects, tasks=tlist))

def exp_content(request, entryid):
	entry = TaskEntry.objects.get(pk=entryid)
	#Create a True/False dictionary for selected features
	sfeats = dict([(f.name, (featlist[f.name].__doc__, f in entry.feats.all())) for f in Feature.objects.all()])
	params = json.loads(entry.params)
	print sfeats

	Exp = experiment.make(tasklist[entry.task.name], feats=[f.name for f in entry.feats.all()])
	traits = Exp.class_traits()
	#traitlen = dict([(name, len(traits[name].default) if isinstance(traits[name].default, (list, tuple)) else 1) for name in Exp.class_editable_traits()])
	traitval = dict([(name, (traits[name].desc, params[name] if name in params else traits[name].default)) for name in Exp.class_editable_traits()])
	return render_to_response("exp_content.html", dict(entry=entry, selected=sfeats, params=traitval, done=True))