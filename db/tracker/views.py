'''
HTML rendering 'view' functions for Django web interface
'''

import json

from django.template import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse

from models import TaskEntry, Task, Subject, Feature, Generator

import namelist
from ajax import exp_tracker

import datetime

def list(request):
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
    td = datetime.timedelta(days=60)
    start_date = datetime.date.today() - td
    entries = TaskEntry.objects.filter(date__gt=start_date).order_by('-date')

    fields = dict(
        entries=entries, 
        subjects=Subject.objects.all().order_by("name"), 
        tasks=Task.objects.filter(visible=True).order_by("name"), 
        features=Feature.objects.filter(visible=True).order_by("name"), 
        generators=Generator.objects.filter(visible=True).order_by("name"),
        hostname=request.get_host(),
        bmi_update_rates=namelist.bmi_update_rates,
        state_spaces=namelist.bmi_state_space_models,
        bmi_algorithms=namelist.bmi_algorithms,
        extractors=namelist.extractors,
        default_extractor=namelist.default_extractor,
        pos_vars=namelist.bmi_training_pos_vars,
    )
    if exp_tracker.task is not None:
        fields['running'] = exp_tracker.task.saveid
    return render_to_response('list.html', fields, RequestContext(request))

def listall(request):
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
    entries = TaskEntry.objects.all().order_by("-date")

    fields = dict(
        entries=entries, 
        subjects=Subject.objects.all().order_by("name"), 
        tasks=Task.objects.filter(visible=True).order_by("name"), 
        features=Feature.objects.filter(visible=True).order_by("name"), 
        generators=Generator.objects.filter(visible=True).order_by("name"),
        hostname=request.get_host(),
        bmi_update_rates=namelist.bmi_update_rates,
        state_spaces=namelist.bmi_state_space_models,
        bmi_algorithms=namelist.bmi_algorithms,
        extractors=namelist.extractors,
        default_extractor=namelist.default_extractor,
        pos_vars=namelist.bmi_training_pos_vars,
    )
    if exp_tracker.task is not None:
        fields['running'] = exp_tracker.task.saveid
    return render_to_response('list.html', fields, RequestContext(request))

def get_sequence(request, idx):
    '''
    Pointing browser to nucleus:8000/sequence_for/(?P<idx>\d+)/ returns a pickled
    file with the 'sequence' used in the specified id
    '''
    import cPickle
    entry = TaskEntry.objects.get(pk=idx)
    seq = cPickle.loads(str(entry.sequence.sequence))
    log = json.loads(entry.report)
    num = len([l[2] for l in log if l[0] == "wait"])

    response = HttpResponse(cPickle.dumps(seq[:num]), content_type='application/x-pickle')
    response['Content-Disposition'] = 'attachment; filename={subj}{time}_{idx}.pkl'.format(
        subj=entry.subject.name[:4].lower(), 
        time="%04d%02d%02d"%(entry.date.year, entry.date.month, entry.date.day),
        idx=idx)
    return response
