import os
import time
import shutil
import json
from SimpleXMLRPCServer import SimpleXMLRPCDispatcher

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from riglib import experiment
from models import TaskEntry, Subject, Calibration, System, DataFile

datapath = "/storage/"

def save_log(idx, log):
    entry = TaskEntry.objects.get(pk=idx)
    entry.report = json.dumps(log)
    entry.save()

def save_calibration(subject, system, name, params):
    print subject, system
    subj = Subject.objects.get(name=subject)
    sys = System.objects.get(name=system)
    Calibration(subject=subj, system=sys, name=name, params=params).save()

def save_data(curfile, system, entry):
    suffix = dict(eyetracker="edf")
    sys = System.objects.get(name=system)
    entry = TaskEntry.objects.get(pk=entry)

    permfile = os.path.join(datapath, system, "{time}.{suff}".format(time.time(), suffix[system]))
    shutil.move(curfile, permfile)
    DataFile(local=True, path=permfile, system=sys, entry=entry).save()

dispatcher = SimpleXMLRPCDispatcher(allow_none=True)
dispatcher.register_function(save_log, 'save_log')
dispatcher.register_function(save_calibration, 'save_cal')

@csrf_exempt
def rpc_handler(request):
    response = HttpResponse(mimetype="application/xml")
    response.write(dispatcher._marshaled_dispatch(request.raw_post_data))
    return response