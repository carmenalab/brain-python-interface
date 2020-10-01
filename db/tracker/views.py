'''
HTML rendering 'view' functions for Django web interface. Retreive data from database to put into HTML format.
'''
import datetime
import pickle
import json
import traceback
import os

from django.template import RequestContext
from django.shortcuts import render
from django.http import HttpResponse

from . import exp_tracker


def main(request):
    return render(request, "main.html", dict())

def _list_exp_history(dbname='default', subject=None, task=None, max_entries=None):
    from . import models
    # from .models import TaskEntry, Task, Subject, Feature, Generator
    td = datetime.timedelta(days=60)

    filter_kwargs = dict(visible=True)
    if not (subject is None) and isinstance(subject, str):
        filter_kwargs['subject__name'] = subject
    if not (task is None) and isinstance(task, str):
        filter_kwargs['task__name'] = task
    
    entries = models.TaskEntry.objects.using(dbname).filter(**filter_kwargs).order_by("-date")
    if isinstance(max_entries, int):
        entries = entries[:max_entries]

    for k in range(0, len(entries)):
        ent = entries[k]
        if k == 0 or not entries[k].date.date() == entries[k-1].date.date():
            ent.html_date = ent.date.date()
        else:
            ent.html_date = None
        ent.html_time = ent.date.time()

    ## Determine how many rows the date should span
    last = -1
    for k, ent in enumerate(entries[::-1]):
        if ent.html_date:
            ent.rowspan = k - last
            last = k

    tasks = models.Task.objects.filter(visible=True).order_by("name")

    epoch = datetime.datetime.utcfromtimestamp(0)
    for entry in entries:
        tdiff = entry.date - epoch
        if tdiff.days % 2 == 0:
            entry.bgcolor = '#E1EEf4'
        else:
            entry.bgcolor = '#FFFFFF'

    subjects = models.Subject.objects.all().order_by("name")
    features = models.Feature.objects.filter(visible=True).order_by("name")
    generators = models.Generator.objects.filter(visible=True).order_by("name")
    collections = models.TaskEntryCollection.objects.all().order_by("name")
    

    fields = dict(
        entries=entries, 
        subjects=subjects, 
        tasks=tasks, 
        features=features, 
        generators=generators,
        collections=collections,
        systems=models.System.objects.all(),
        n_blocks=len(entries),
    )

    try:
        from config import bmiconfig
        fields['bmi_update_rates'] = bmiconfig.bmi_update_rates
        fields['state_spaces'] = bmiconfig.bmi_state_space_models
        fields['bmi_algorithms'] = bmiconfig.bmi_algorithms
        fields['extractors'] = bmiconfig.extractors
        fields['default_extractor'] = bmiconfig.default_extractor
        fields['pos_vars'] = bmiconfig.bmi_training_pos_vars      # 'pos_vars' indicates which column of the task HDF table to look at to extract kinematic data 
        fields['kin_extractors'] = bmiconfig.kin_extractors       # post-processing methods on the selected kinematic variable
    except ImportError:
        pass
    return fields

def list_exp_history(request, **kwargs):
    '''
    Top-level view called when browser pointed at webroot

    Parameters
    ----------
    request: HTTPRequest instance
        No data needs to be extracted from this request

    Returns 
    -------
    Django HTTPResponse instance
    '''
    from .models import TaskEntry, Task, Subject, Feature, Generator

    fields = _list_exp_history(**kwargs)
    fields['hostname'] = request.get_host()

    # this line is important--this is needed so the Track object knows if the task has ended in an error
    # TODO there's probably some better way of doing this within the multiprocessing lib (some code to run after the process has terminated)
    tracker = exp_tracker.get()
    tracker.update_alive()

    if tracker.task_proxy is not None and "saveid" in tracker.task_kwargs:
        fields['running'] = tracker.task_kwargs["saveid"]

    resp = render(None,'list.html', fields)
    return resp

def setup(request):
    from . import models
    from .models import TaskEntry, Task, Subject, Feature, Generator

    subjects = models.Subject.objects.all()
    tasks = models.Task.objects.all()
    features = models.Feature.objects.all()
    systems = models.System.objects.all()

    # populate the list of built-in features which could be added
    from features import built_in_features
    built_in_feature_names = list(built_in_features.keys())
    active_features = []
    for feat in features:
        if feat.name in built_in_feature_names:
            built_in_feature_names.remove(feat.name)
            active_features.append(feat)

    # list of available databases
    from db import settings
    databases = list(settings.DATABASES.keys())

    # # check that a 'database file' system exists
    # db_sys = models.System.objects.filter(name="metadata")
    # if len(db_sys) == 0:
    #     db_sys = models.System(name="metadata", path="db")
    #     db_sys.save()

    database_objs = []
    for db in databases:
        path = models.KeyValueStore.get('data_path', '--', dbname=db)
        database_objs.append({"name":db, "path":path})

        # database_obj = models.DataFile.objects.filter(path='db_%s.sql' % db)
        # if len(database_obj) == 0:
        #     database_objs.append({"name":db, "path":"--"})
        # else:
        #     database_obj = database_obj[0]
        #     database_obj.name = db
        #     database_objs.append(database_obj)

    print(database_objs)

    recording_sys = models.KeyValueStore.get("recording_sys")
    recording_sys_options = ['None', 'tdt', 'blackrock', 'plexon']

    return render(request, "setup.html", 
        dict(subjects=subjects, tasks=tasks, systems=systems, active_features=active_features,
            built_in_feature_names=built_in_feature_names, databases=database_objs,
            recording_sys=recording_sys, recording_sys_options=recording_sys_options))

def _color_entries(entries):
    from .models import TaskEntry, Task, Subject, Feature, Generator

    epoch = datetime.datetime.utcfromtimestamp(0)
    
    last_tdiff = entries[0].date - epoch
    colors = ['#E1EEf4', '#FFFFFF']
    color_idx = 0
    for entry in entries:
        tdiff = entry.date - epoch
        if not (tdiff.days == last_tdiff.days):
            color_idx = (color_idx + 1) % 2
            last_tdiff = tdiff
        entry.bgcolor = colors[color_idx]

def get_sequence(request, idx):
    '''
    Pointing browser to WEBROOT/sequence_for/(?P<idx>\d+)/ returns a pickled
    file with the 'sequence' used in the specified id
    '''
    from .models import TaskEntry, Task, Subject, Feature, Generator
    entry = TaskEntry.objects.get(pk=idx)
    seq = pickle.loads(str(entry.sequence.sequence))
    log = json.loads(entry.report)
    num = len([l[2] for l in log if l[0] == "wait"])

    response = HttpResponse(pickle.dumps(seq[:num]), content_type='application/x-pickle')
    response['Content-Disposition'] = 'attachment; filename={subj}{time}_{idx}.pkl'.format(
        subj=entry.subject.name[:4].lower(), 
        time="%04d%02d%02d"%(entry.date.year, entry.date.month, entry.date.day),
        idx=idx)
    return response

def link_data_files_view_generator(request, task_entry_id):
    from . import models
    from .models import TaskEntry, Task, Subject, Feature, Generator
    systems = models.System.objects.all()
    display_data = dict(systems=systems, task_entry_id=task_entry_id)
    return render(request, "link_data_files.html", display_data)

from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def link_data_files_response_handler(request, task_entry_id):
    from . import models
    file_path = request.POST["file_path"]
    raw_data = request.POST["raw_data"]
    raw_data_format = request.POST['raw_data_format']
    data_system_id = request.POST["data_system_id"]
    
    task_entry = models.TaskEntry.objects.get(id=task_entry_id)
    system = models.System.objects.get(id=data_system_id)
    if not os.path.isabs(file_path):
        sys_path = system.path
        if system.input_path is not None and system.input_path != "":
            sys_path = system.input_path
        else:
            sys_path = system.path
        file_path = os.path.join(sys_path, file_path)
    
    print("Adding file", file_path)

    if len(raw_data.strip()) > 0:
        # new file
        try:        
            with open(file_path, 'w') as fp:
                fp.write(raw_data)
        except:
            import traceback
            traceback.print_exc()
            return HttpResponse("Unable to add new file")

    if os.path.exists(file_path):
        try:
            data_file = models.DataFile.create(system, task_entry, file_path, local=True, archived=False)
            return HttpResponse("Added new data file: {}".format(data_file.path))
        except:
            print("Error creating data file")
            traceback.print_exc()
            return HttpResponse("Error creating data file, see terminal for error message")
    else:
        return HttpResponse("File does not exist {}!".format(file_path))






