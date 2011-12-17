import json

from django.http import HttpResponse

import views

import tasks
from riglib import experiment

def exp_info(request, taskname):
	base = tasks.tasklist[taskname]
	Exp = experiment.make(base, feats=[k for k,v in request.GET.items() if v])
	traits = Exp.class_traits()
	data = dict([ (name, (traits[name].desc, traits[name].default)) for name in Exp.class_editable_traits()])
	return HttpResponse(json.dumps(data), mimetype="application/json")