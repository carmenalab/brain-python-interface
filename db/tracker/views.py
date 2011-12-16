import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))

from django.http import HttpResponse

from tracker.models import TaskEntry

from riglib import experiment
from tasks import RDS

expqueue = []

def start_task(request):
	Exp = experiment.make(Dots, feats=("autostart", "button"))
	expqueue.append(Exp)
	#Exp().start()
	return HttpResponse("Bleh")

def get_info(request):
	return HttpResponse(expqueue[0].class_editable_traits())