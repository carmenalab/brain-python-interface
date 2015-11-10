'''
Methods for remotely interacting with the sqlite3 database using remote procedure call (RPC)
For example, linking HDF file to a particular task entry.
'''

import os
import time
import json
import shutil
import datetime
from SimpleXMLRPCServer import SimpleXMLRPCDispatcher

import django
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from distutils.version import StrictVersion

from riglib import experiment
from models import TaskEntry, Subject, Calibration, System, DataFile, Decoder
import cPickle
import tempfile

def save_log(idx, log, dbname='default'):
    entry = TaskEntry.objects.using(dbname).get(pk=idx)
    entry.report = json.dumps(log)
    entry.save()

def save_calibration(subject, system, name, params, dbname='default'):
    print subject, system
    subj = Subject.objects.using(dbname).get(name=subject)
    sys = System.objects.using(dbname).get(name=system)
    Calibration(subject=subj, system=sys, name=name, params=params).save(using=dbname)

def save_data(curfile, system, entry, move=True, local=True, custom_suffix=None, dbname='default'):
    suffix = dict(eyetracker="edf", hdf="hdf", plexon="plx", bmi="pkl", bmi_params="npz", juice_log="png", video="avi")
    if system in suffix:
        suff = suffix[system]
    else:  # blackrock system saves multiple files (.nev, .ns1, .ns2, etc.)
        if custom_suffix is not None:
            suff = custom_suffix
        else:
            raise Exception('Must provide a custom suffix for system: ' + system)

    sys = System.objects.using(dbname).get(name=system)
    entry = TaskEntry.objects.using(dbname).get(pk=entry)

    now = entry.date
    today = datetime.date(now.year, now.month, now.day)
    tomorrow = today + datetime.timedelta(days=1)

    entries = TaskEntry.objects.using(dbname).filter(date__gte=today, date__lte=tomorrow)
    enums = dict([(e, i) for i, e in enumerate(entries.order_by("date"))])
    num = enums[entry]

    if move:
        dataname = "{subj}{time}_{num:02}_te{id}.{suff}".format(
            subj=entry.subject.name[:4].lower(),
            time=time.strftime('%Y%m%d'), num=num+1,
            id=entry.id, suff=suff
        )
        fullname = os.path.join(sys.path, dataname)
        permfile = dataname

        if os.path.abspath(sys.path) == os.path.abspath(os.path.split(curfile)[0]):
            print "moving file..."
            os.rename(curfile, fullname)
        elif not os.path.exists(fullname):
            print "copying file..."
            shutil.copy2(curfile, os.path.join(sys.path, dataname))
        else:
            raise ValueError('Will not overwrite existing files')
    else:
        permfile = curfile

    DataFile(local=local, path=permfile, system=sys, entry=entry).save(using=dbname)
    print "Saved datafile for file=%s -> %s, system=%s, id=%d)..." % (curfile, permfile, system, entry.id)

def save_bmi(name, entry, filename, dbname='default'):
    '''
    Save BMI objects to database
    '''
    entry = TaskEntry.objects.using(dbname).get(pk=entry)
    now = entry.date
    today = datetime.date(now.year, now.month, now.day)
    tomorrow = today + datetime.timedelta(days=1)

    entries = TaskEntry.objects.using(dbname).filter(date__gte=today, date__lte=tomorrow)
    enums = dict([(e, i) for i, e in enumerate(entries.order_by("date"))])
    num = enums[entry]

    pklname = "{subj}{time}_{num:02}_{name}.pkl".format(
        subj=entry.subject.name[:4].lower(),
        time=entry.date.strftime('%Y%m%d'),
        num=num, name=name)
    base = System.objects.using(dbname).get(name='bmi').path
    shutil.copy2(filename, os.path.join(base, pklname))

    Decoder(name=name,entry=entry,path=pklname).save(using=dbname)
    print "Saved decoder to %s"%os.path.join(base, pklname)


#############################################################################
##### Register functions for remote procedure call from other processes #####
#############################################################################
dispatcher = SimpleXMLRPCDispatcher(allow_none=True)
dispatcher.register_function(save_log, 'save_log')
dispatcher.register_function(save_calibration, 'save_cal')
dispatcher.register_function(save_data, 'save_data')
dispatcher.register_function(save_bmi, 'save_bmi')

@csrf_exempt
def rpc_handler(request):
    if StrictVersion('%d.%d.%d' % django.VERSION[0:3]) < StrictVersion('1.6.0'):
        response = HttpResponse(mimetype="application/xml") 
        response.write(dispatcher._marshaled_dispatch(request.raw_post_data))
    else:
        response = HttpResponse(mimetype="application/xml") 
        response.write(dispatcher._marshaled_dispatch(request.body))
    return response
