import json

from django.template import RequestContext
from django.shortcuts import render_to_response

from models import TaskEntry, Task, Subject, Feature, Generator

import namelist
from riglib import experiment
from ajax import display

def list(request):
	return render_to_response('list.html', dict(
		entries=TaskEntry.objects.all().order_by("-date")[:100], 
		subjects=Subject.objects.all().order_by("name"), 
		tasks=Task.objects.all().order_by("name"), 
		features=Feature.objects.order_by("name").all(), 
		generators=Generator.objects.order_by("name").all()
	), RequestContext(request))