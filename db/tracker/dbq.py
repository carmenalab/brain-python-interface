'''
Methods for remotely interacting with the sqlite3 database using remote procedure call (RPC)
For example, linking HDF file to a particular task entry.
'''

import os
import time
import json
import shutil
import datetime
from xmlrpc.server import SimpleXMLRPCDispatcher

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .models import TaskEntry, Subject, Calibration, System, DataFile, Decoder

def save_log(idx, log, dbname='default'):
    entry = TaskEntry.objects.using(dbname).get(pk=idx)
    entry.report = json.dumps(log)
    entry.save()

def save_calibration(subject, system, name, params, dbname='default'):
    print(subject, system)
    subj = Subject.objects.using(dbname).get(name=subject)
    sys = System.objects.using(dbname).get(name=system)
    Calibration(subject=subj, system=sys, name=name, params=params).save(using=dbname)

def save_data(curfile, system, entry, move=True, local=True, custom_suffix=None, dbname='default'):
    suffix = dict(supp_hdf="supp.hdf", eyetracker="edf", hdf="hdf", plexon="plx", bmi="pkl", bmi_params="npz", juice_log="png", video="avi", optitrack="tak")
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

        # Set the new filename and filepath
        permfile = "{subj}{time}_{num:02}_te{id}.{suff}".format(
            subj=entry.subject.name[:4].lower(),
            time=time.strftime('%Y%m%d'), num=num+1,
            id=entry.id, suff=suff
        )
        if system == 'blackrock2':
            fullname = os.path.join('/storage/rawdata/blackrock', permfile)
            sys_path = '/storage/rawdata/blackrock'
            print('BLACKROCK SYSTEM: ')
        else:
            fullname = os.path.join(sys.path, permfile)
            sys_path = sys.path

        # Move or copy or rsync the file
        if os.path.abspath(sys_path) == os.path.abspath(os.path.split(curfile)[0]):
            print("moving file...")
            os.rename(curfile, fullname) # from sys_path to sys_path
        elif os.path.exists(sys_path) and not os.path.exists(fullname):
            print("copying file...")
            shutil.copy2(curfile, os.path.join(sys_path, permfile)) # from somewhere to sys_path
        else:
            raise ValueError('Will not overwrite existing files')
    else:
        permfile = curfile

    DataFile(local=local, path=permfile, system=sys, entry=entry).save(using=dbname)
    print("Saved datafile for file=%s -> %s, system=%s, id=%d)..." % (curfile, permfile, system, entry.id))

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

    #Make sure decoder name doesn't exist already:
    #Make sure new decoder name doesn't already exist:
    import os.path
    dec_ix = 0

    while os.path.isfile(os.path.join(base, pklname)):
        pklname = "{subj}{time}_{num:02}_{name}_{ix}.pkl".format(
        subj=entry.subject.name[:4].lower(),
        time=entry.date.strftime('%Y%m%d'),
        num=num, name=name,ix=dec_ix)
        dec_ix += 1

    # Some problems with this on windows with permissions
    shutil.copy2(filename, os.path.join(base, pklname))

    Decoder(name=name,entry=entry,path=pklname).save(using=dbname)
    # try:
    #     decoder_entry = Decoder.objects.using(dbname).get(entry=entry)
    # except:
    #     print('too many decoders to list: ')
    #     import dbfunctions as dbfn
    #     d = dbfn.TaskEntry(entry.pk)
    #     d_list = d.get_decoders_trained_in_block()
    #     for d in d_list:
    #         print(f"{d.pk}, {d.name}")
    print("Saved decoder to %s"%os.path.join(base, pklname))

def cleanup(entry, dbname='default'):
    '''
    Final cleanup after a task is finished
    '''
    te = TaskEntry.objects.using(dbname).get(id=entry)
    tries = 3
    while tries > 0:
        if te.make_hdf_self_contained():
            break
        time.sleep(1) # wait for the hdf file to be created
        tries -= 1

def hide_task_entry(entry, dbname='default'):
    te = TaskEntry.objects.using(dbname).get(id=entry)
    te.visible = False
    te.save()



#############################################################################
##### Register functions for remote procedure call from other processes #####
#############################################################################
dispatcher = SimpleXMLRPCDispatcher(allow_none=True)
dispatcher.register_function(save_log, 'save_log')
dispatcher.register_function(save_calibration, 'save_cal')
dispatcher.register_function(save_data, 'save_data')
dispatcher.register_function(save_bmi, 'save_bmi')
dispatcher.register_function(cleanup, 'cleanup')
dispatcher.register_function(hide_task_entry, 'hide_task_entry')

@csrf_exempt
def rpc_handler(request):
    response = HttpResponse(content_type="application/xml")
    response.write(dispatcher._marshaled_dispatch(request.body))
    return response
