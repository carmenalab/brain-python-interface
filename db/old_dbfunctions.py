'''
Deprecated functionality from dbfunctions
'''

import os
import sys
import json
import numpy as np
import datetime
import pickle
import cPickle
import db.paths
import tables
import matplotlib.pyplot as plt
import time, datetime
from scipy.stats import nanmean
from collections import defaultdict, OrderedDict

try:
    import plotutil
except:
    pass

def get_param(entry, name):
    '''
    Returns the value of a specific parameter in the param list. Takes
    TaskEntry object and string for exact param name.
    '''
    params = get_params(entry)
    return params[name]

def get_completed_trials(entry):
    '''
    Returns # of trials rewarded.
    Takes TaskEntry object.
    '''
    report = json.loads(entry.report)
    count1=0.0
    for s in report:
        if s[0]=='reward':
            count1+=1
    return count1

def get_length(entry):
    '''
    Returns length of session in seconds.
    Takes TaskEntry object.
    '''
    report = json.loads(entry.report)
    return report[-1][2]-report[0][2]


def search_by_date(date, subj=None):
    '''
    Get all the task entries for a particular date
    '''
    kwargs = dict(date__year=date.year, date__month=date.month, 
                  date__day=date.day)
    if isinstance(subj, str) or isinstance(subj, unicode):
        kwargs['subject__name__startswith'] = str(subj)
    elif subj is not None:
        kwargs['subject__name'] = subj.name
    return models.TaskEntry.objects.filter(**kwargs)

def get_task_entry(entry_id):
    '''
    Returns the task entry object from the database with the specified entry_id.
    entry_id = int
    '''
    return models.TaskEntry.objects.get(pk=entry_id)

def get_task_id(name):
    '''
    Returns the task ID for the specified task name.
    '''
    return models.Task.objects.get(name=name).pk


def get_decoder_entry(entry):
    '''Returns the database entry for the decoder used in the session. Argument can be a task entry
    or the ID number of the decoder entry itself.
    '''
    if isinstance(entry, int):
        try:
            return models.Decoder.objects.get(pk=entry)
        except:
            return None
    else:
        params = json.loads(entry.params)
        pk = None
        if 'decoder' in params:
            pk = params['decoder']
        elif 'bmi' in params:
            pk = params['bmi']
        else:
            return None

        try:
            return models.Decoder.objects.get(pk=pk)
        except:
            return None

def get_decoder_name(entry):
    ''' 
    Returns the filename of the decoder used in the session.
    Takes TaskEntry object.
    '''
    entry = lookup_task_entries(entry)
    try:
        decid = json.loads(entry.params)['decoder']
    except:
        decid = json.loads(entry.params)['bmi']
    return models.Decoder.objects.get(pk=decid).path

def get_decoder_name_full(entry):
    entry = lookup_task_entries(entry)
    decoder_basename = get_decoder_name(entry)
    return os.path.join(paths.data_path, 'decoders', decoder_basename)

def get_decoder(entry):
    entry = lookup_task_entries(entry)
    filename = get_decoder_name_full(entry)
    dec = pickle.load(open(filename, 'r'))
    dec.db_entry = get_decoder_entry(entry)
    dec.name = dec.db_entry.name
    return dec

def get_params(entry):
    '''
    Returns a dict of all task params for session.
    Takes TaskEntry object.
    '''
    return json.loads(entry.params)

def get_task_name(entry):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return models.Task.objects.get(pk=entry.task_id).name
    
def get_date(entry):
    '''
    Returns date and time of session (as a datetime object).
    Takes TaskEntry object.
    '''
    return entry.date
    
def get_notes(entry):
    '''
    Returns notes for session.
    Takes TaskEntry object.
    '''
    return entry.notes
    
def get_subject(entry):
    '''
    Returns name of subject for session.
    Takes TaskEntry object.
    '''
    return models.Subject.objects.get(pk=entry.subject_id).name
    


class TaskMessages(object):
    def __init__(self, *task_msgs):
        self.task_msgs = np.hstack(task_msgs)

    def __getattr__(self, attr):
        if attr == 'time':
            return self.task_msgs['time']
        elif attr == 'msg':
            return self.task_msgs['msg']
        else:
            return TaskMessages(self.task_msgs[self.task_msgs['msg'] == attr])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.task_msgs[idx]
        elif isinstance(idx, str) or isinstance(idx, unicode):
            return self.__getattr__(idx)
        elif np.iterable(idx):
            task_msgs = np.hstack([self.__getitem__(i).task_msgs for i in idx])
            task_msg_inds = np.argsort(task_msgs['time'])
            return TaskMessages(task_msgs[task_msg_inds])
            # return np.hstack([self.__getitem__(i) for i in idx])


