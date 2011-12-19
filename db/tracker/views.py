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
	featdoc = dict([(f.name, featlist[f.name].__doc__) for f in Feature.objects.all()])
	return render_to_response('list.html', dict(entries=entries, subjects=subjects, tasks=tlist, feats=featdoc))

def exp_content(request, entryid):
	entry = TaskEntry.objects.get(pk=entryid)
	#Create a True/False dictionary for selected features
	
	return render_to_response("exp_content.html", dict(entry=entry, selected=sfeats, params=traitval, done=True))