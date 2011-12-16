import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))
import json

from django.http import HttpResponse

from tracker.models import TaskEntry, Task

from riglib import experiment
import tasks

expqueue = []

def start_task(request):
	Exp = experiment.make(Dots, feats=("autostart", "button"))
	expqueue.append(Exp)
	#Exp().start()
	return HttpResponse("Bleh")

def exp_info(request, taskname):
	base = tasks.tasklist[taskname]
	Exp = experiment.make(base, feats=[k for k,v in request.GET.items() if v])
	traits = Exp.class_traits()
	data = dict([ (name, (traits[name].desc, traits[name].default)) for name in Exp.class_editable_traits()])
	return HttpResponse(json.dumps(data), mimetype="application/json")