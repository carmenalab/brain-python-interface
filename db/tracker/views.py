import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))

from django.shortcuts import render_to_response

from tracker.models import TaskEntry, Task, Subject

expqueue = []

def start_task(request):
	Exp = experiment.make(Dots, feats=("autostart", "button"))
	expqueue.append(Exp)

def list(request):
	entries = TaskEntry.objects.all().order_by("-date")[:100]
	tlist = Task.objects.all().order_by("name")
	subjects = Subject.objects.all().order_by("name")
	return render_to_response('list.html', dict(entries=entries, subjects=subjects, tasks=tlist))