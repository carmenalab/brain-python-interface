import os
import time
import json
import shutil
import datetime
from SimpleXMLRPCServer import SimpleXMLRPCDispatcher

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from riglib import experiment
from models import TaskEntry, Subject, Calibration, System, DataFile, Decoder

def save_log(idx, log):
    entry = TaskEntry.objects.get(pk=idx)
    entry.report = json.dumps(log)
    entry.save()

def save_calibration(subject, system, name, params):
    print subject, system
    subj = Subject.objects.get(name=subject)
    sys = System.objects.get(name=system)
    Calibration(subject=subj, system=sys, name=name, params=params).save()

def save_data(curfile, system, entry, move=True, local=True):
    suffix = dict(eyetracker="edf", hdf="hdf", plexon="plx")
    sys = System.objects.get(name=system)
    entry = TaskEntry.objects.get(pk=entry)

    now = entry.date
    today = datetime.date(now.year, now.month, now.day)
    tomorrow = today + datetime.timedelta(days=1)

    entries = TaskEntry.objects.filter(date__gte=today, date__lte=tomorrow)
    enums = dict([(e, i) for i, e in enumerate(entries.order_by("date"))])
    num = enums[entry]

    if move:
        dataname = "{subj}{time}_{num:02}.{suff}".format(
            subj=entry.subject.name[:4].lower(),
            time=time.strftime('%Y%m%d'), num=num+1,
            suff=suffix[system]
        )
        permfile = dataname
        if sys.path == os.path.split(curfile)[0]:
            os.rename(curfile, os.path.join(sys.path, dataname))
        else:
            shutil.move(curfile, os.path.join(sys.path, dataname))
    else:
        permfile = curfile

    DataFile(local=local, path=permfile, system=sys, entry=entry).save()
    print "Saved datafile for file=%s -> %s, system=%s, id=%d)..."%(curfile, permfile, system, entry.id)

def save_bmi(name, entry, filename):
    entry = TaskEntry.objects.get(pk=entry)
    now = entry.date
    today = datetime.date(now.year, now.month, now.day)
    tomorrow = today + datetime.timedelta(days=1)

    entries = TaskEntry.objects.filter(date__gte=today, date__lte=tomorrow)
    enums = dict([(e, i) for i, e in enumerate(entries.order_by("date"))])
    num = enums[entry]

    pklname = "{subj}{time}_{num:02}_{name}.pkl".format(
        subj=entry.subject.name[:4].lower(),
        time=entry.date.strftime('%Y%m%d'),
        num=num, name=name)
    base = System.objects.get(name='bmi').path
    shutil.copy2(filename, os.path.join(base, pklname))

    Decoder(name=name,entry=entry,path=pklname).save()
    print "Saved decoder to %s"%os.path.join(base, pklname)

def entry_error(entry):
    TaskEntry.objects.get(pk=entry).remove()
    print "Removed Bad Entry %d"%entry

dispatcher = SimpleXMLRPCDispatcher(allow_none=True)
dispatcher.register_function(save_log, 'save_log')
dispatcher.register_function(save_calibration, 'save_cal')
dispatcher.register_function(save_data, 'save_data')
dispatcher.register_function(save_bmi, 'save_bmi')
dispatcher.register_function(entry_error, 'entry_error')

@csrf_exempt
def rpc_handler(request):
    response = HttpResponse(mimetype="application/xml")
    response.write(dispatcher._marshaled_dispatch(request.raw_post_data))
    return response
