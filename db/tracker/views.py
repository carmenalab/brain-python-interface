'''
HTML rendering 'view' functions for Django web interface
'''

import json

from django.template import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse

from models import TaskEntry, Task, Subject, Feature, Generator

import namelist
from ajax import display

def list(request):
    '''
    Top-level view called when browser pointed at nucleus:8000/
    '''
    fields = dict(
        entries=TaskEntry.objects.all().order_by("-date"), 
        subjects=Subject.objects.all().order_by("name"), 
        tasks=Task.objects.filter(visible=True).order_by("name"), 
        features=Feature.objects.filter(visible=True).order_by("name"), 
        generators=Generator.objects.filter(visible=True).order_by("name"),
        hostname=request.get_host(),
        bmis=namelist.bmis,
        extractors=namelist.extractors,
    )
    if display.task is not None:
        fields['running'] = display.task.saveid
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
