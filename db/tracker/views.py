import json

from django.template import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse

from models import TaskEntry, Task, Subject, Feature, Generator

import namelist
from riglib import experiment
from ajax import display

def list(request):
    fields = dict(
        entries=TaskEntry.objects.all().order_by("-date")[:50], 
        subjects=Subject.objects.all().order_by("name"), 
        tasks=Task.objects.all().order_by("name"), 
        features=Feature.objects.order_by("name").all(), 
        generators=Generator.objects.order_by("name").all(),
        hostname=request.get_host(),
    )
    if display.task is not None:
        fields['running'] = display.task.saveid
    return render_to_response('list.html', fields, RequestContext(request))

def get_sequence(request, idx):
    import cPickle
    entry = TaskEntry.objects.get(pk=idx)
    seq = cPickle.loads(str(entry.sequence.sequence))
    log = json.loads(entry.report)
    num = len([l[2] for l in log if l[0] == "wait"])

    response = HttpResponse(cPickle.dumps(seq[:num]), content_type='application/x-pickle')
    response['Content-Disposition'] = 'attachment; filename={subj}{time}.pkl'.format(
        subj=entry.subject.name[:4].lower(), 
        time="%04d%02d%02d"%(entry.date.year, entry.date.month, entry.date.day))
    return response
