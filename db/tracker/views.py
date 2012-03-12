import json

from django.template import RequestContext
from django.shortcuts import render_to_response

from models import TaskEntry, Task, Subject, Feature, Generator

import namelist
from riglib import experiment
from ajax import display

def list(request):
	entries = TaskEntry.objects.all().order_by("-date")[:100]
	tlist = Task.objects.all().order_by("name")
	subjects = Subject.objects.all().order_by("name")
	featdoc = dict([(f.name, namelist.features[f.name].__doc__) for f in Feature.objects.all()])
	gens = Generator.objects.all().order_by("name")
	jsongens = json.dumps(dict([(g.id, (g.name, g.params, g.static)) for g in gens]))

	return render_to_response('list.html', dict(
		running=display.expidx if display.state == "running" else None,
		entries=entries, 
		subjects=subjects, 
		tasks=tlist, 
		feats=featdoc, 
		gens=jsongens),
		RequestContext(request))