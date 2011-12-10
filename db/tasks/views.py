from tasks.models import TaskEntry
import os
import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, "..", ".."))
from riglib import experiment
from riglib.tasks import Dots

def start_task(request):
	Exp = experiment.make_experiment(Dots, feats=("autostart", "button"))
	Exp().start()
	return ""