import os
import sys
import json
import numpy as np
import datetime

os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
sys.path.append(os.path.expanduser("~/code/bmi3d/db/"))
from tracker import models

def get_task_entry(entry_id):
    '''
    Returns the task entry object from the database with the specified entry_id.
    entry_id = int
    '''
    return models.TaskEntry.objects.get(pk=entry_id)

def get_decoder_entry(entry):
	''' 
	Returns the filename of the decoder used in the session.
    Takes TaskEntry object.
	'''

	params = json.loads(entry.params)
	if 'bmi' in params:
		return models.Decoder.objects.get(pk=params['bmi'])
	else:
		return None

def get_decoder_name(entry):
	''' 
	Returns the filename of the decoder used in the session.
    Takes TaskEntry object.
	'''
	decid = json.loads(entry.params)['bmi']
	return models.Decoder.objects.get(pk=decid).path

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
	
def get_length(entry):
    '''
    Returns length of session in seconds.
    Takes TaskEntry object.
    '''
    report = json.loads(entry.report)
    return report[-1][2]-report[0][2]
    
def get_success_rate(entry):
    '''
    Returns (# of trials rewarded)/(# of trials intiated).
    Takes TaskEntry object.
    '''
    report = json.loads(entry.report)
    count1=0.0
    count2=0.0
    for s in report:
        if s[0]=='reward':
            count1+=1
        if s[0]=='wait':
            count2+=1
    return count1/count2

def get_initiate_rate(entry):
    '''
    Returns average # of trials initated per minute.
    Takes TaskEntry object.
    '''
    length = get_length(entry)
    report = json.loads(entry.report)
    count=0.0
    for s in report:
        if s[0]=='wait':
            count+=1
    return count/(length/60.0)

def get_reward_rate(entry):
    '''
    Returns average # of trials completed per minute.
    Takes TaskEntry object.
    '''
    length = get_length(entry)
    report = json.loads(entry.report)
    count=0.0
    for s in report:
        if s[0]=='reward':
            count+=1
    return count/(length/60.0)
    
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
    
def get_param(entry, name):
    '''
    Returns the value of a specific parameter in the param list. Takes
    TaskEntry object and string for exact param name.
    '''
    params = get_params(entry)
    return params[name]
    
def session_summary(entry):
    '''
    Prints a summary of info about session.
    Takes TaskEntry object.
    '''
    print "Subject: ", get_subject(entry)
    print "Task: ", get_task_name(entry)
    print "Date: ", str(get_date(entry))
    hours = np.floor(get_length(entry)/3600)
    mins = np.floor(get_length(entry)/60) - hours*60
    secs = get_length(entry) - mins*60
    print "Length: " + str(int(hours))+ ":" + str(int(mins)) + ":" + str(int(secs))
    try:
        print "Assist level: ", get_param(entry,'assist_level')
    except:
        print "Assist level: 0"
    print "Completed trials: ", get_completed_trials(entry)
    print "Success rate: ", get_success_rate(entry)*100, "%"
    print "Reward rate: ", get_reward_rate(entry), "trials/minute"
    print "Initiate rate: ", get_initiate_rate(entry), "trials/minute"
    
def query_daterange(startdate, enddate=datetime.date.today()):
    '''
    Returns QuerySet for task entries within date range (inclusive). startdate and enddate
    are date objects. End date is optional- today's date by default.
    '''
    return models.TaskEntry.objects.filter(date__gte=startdate).filter(date__lte=enddate)
    
def get_hdf_file(entry):
    '''
    Returns the name of the hdf file associated with the session.
    '''
    hdf = models.System.objects.get(name='hdf')
    q = models.DataFile.objects.filter(entry_id=entry.id).filter(system_id=hdf.id)
    if len(q)==0:
        return None
    else:
        try:
            import db.paths
            return os.path.join(db.paths.data_path, hdf.name, q[0].path)
        except:
            return q[0].path

def get_plx_file(entry):
    '''
    Returns the name of the plx file associated with the session.
    '''
    plexon = models.System.objects.get(name='plexon')
    q = models.DataFile.objects.filter(entry_id=entry.id).filter(system_id=plexon.id)
    if len(q)==0:
        return None
    else:
        try:
            import db.paths
            return os.path.join(db.paths.data_path, plexon.name, q[0].path)
        except:
            return q[0].path
        
def get_plx2_file(entry):
    '''
    Returns the name of the plx2 file associated with the session.
    '''
    plexon = models.System.objects.get(name='plexon2')
    q = models.DataFile.objects.filter(entry_id=entry.id).filter(system_id=plexon.id)
    if len(q)==0:
        return None
    else:
        try:
            import db.paths
            return os.path.join(db.paths.data_path, plexon.name, q[0].path)
        except:
            return q[0].path
        
def get_bmiparams_file(entry):
    '''
    Returns the name of the bmi parameter update history file associated with the session.
    '''
    bmi_params = models.System.objects.get(name='bmi_params')
    q = models.DataFile.objects.filter(entry_id=entry.id).filter(system_id=bmi_params.id)
    if len(q)==0:
        return None
    else:
    	try:
    		import db.paths
    		return os.path.join(db.paths.data_path, bmi_params.name, q[0].path)
    	except:
        	return q[0].path


def get_decoder_parent(decoder):
	'''
	decoder = database record of decoder object
	'''
	entryid = decoder.entry_id
	te = get_task_entry(entryid)
	return get_decoder_entry(te)

def get_decoder_sequence(decoder):
	'''
	decoder = database record of decoder object
	'''	
	parent = get_decoder_parent(decoder)
	if parent is None:
		return [None]
	else:
		return [parent] + get_decoder_sequence(parent)