import os
import sys
import json
import numpy as np
import datetime
import cPickle
import db.paths
import tables
import matplotlib.pyplot as plt
from scipy.stats import nanmean

os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
sys.path.append(os.path.expanduser("~/code/bmi3d/db/"))
from tracker import models
from db import paths

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
        return models.Decoder.objects.get(pk=entry)
    else:
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

def get_decoder_name_full(entry):
    decoder_basename = get_decoder_name(entry)
    return os.path.join(paths.data_path, 'decoders', decoder_basename)

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
            return os.path.join(db.paths.rawdata_path, hdf.name, q[0].path)
        except:
            return q[0].path

def get_hdf(entry):
    '''
    Return hdf opened file
    '''
    hdf_filename = get_hdf_file(entry)
    hdf = tables.openFile(hdf_filename)
    return hdf

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
        return [decoder]
    else:
        return [decoder] + get_decoder_sequence(parent)

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

def search_by_decoder(decoder):
    '''
    Returns task entries that used specified decoder. Decoder argument can be
    decoder entry or entry ID.
    '''
    if isinstance(decoder, int):
        decid = decoder
    else:
        decid = decoder.id
    return models.TaskEntry.objects.filter(params__contains='"bmi": '+str(decid))

def search_by_units(unitlist, decoderlist = None, exact=False):
    '''
    Returns decoder entries that contain the specified units. If exact is True,
    returns only decoders whose unit lists match unitlist exactly, otherwise
    returns decoders that contain units in unitlist in addition to others. If
    given a list of decoder entries, only searches within those, otherwise searches
    all decoders in database.
    '''
    if decoderlist is not None:
        all_decoders = decoderlist
    else:
        all_decoders = models.Decoder.objects.all()
    subset = set(tuple(unit) for unit in unitlist)
    dec_list = []
    for dec in all_decoders:
        try:
            decobj = cPickle.load(open(db.paths.data_path+'/decoders/'+dec.path))
            decset = set(tuple(unit) for unit in decobj.units)
            if subset==decset:
                dec_list = dec_list + [dec]
            elif not exact and subset.issubset(decset):
                dec_list = dec_list + [dec]
        except:
            pass
    return dec_list



def get_code_version():
    import os
    git_version_hash = os.popen('bmi3d_git_hash').readlines()
    git_version_hash = git_version_hash[0].rstrip('\n')
    return git_version_hash

def get_rewards_per_min(task_entry, window_size_mins=1.):
    '''
    Estimates rewards per minute. New estimates are made every 1./60 seconds
    using the # of rewards observed in the previous 'window_size_mins' minutes 
    '''
    hdf_filename = get_hdf_file(task_entry)
    hdf = tables.openFile(hdf_filename)
    task_msgs = hdf.root.task_msgs[:]
    reward_msgs = filter(lambda m: m[0] == 'reward', task_msgs)
    reward_on = np.zeros(hdf.root.task.shape)
    for reward_msg in reward_msgs:
        reward_on[reward_msg[1]] = 1
    conv = np.ones(window_size_mins * 3600) * 1./window_size_mins
    rewards_per_min = np.convolve(reward_on, conv, 'valid')
    return rewards_per_min

def plot_rewards_per_min(task_entry, show=False, **kwargs):
    '''
    Make a plot of the rewards per minute
    '''
    rewards_per_min = get_rewards_per_min(task_entry, **kwargs)
    plt.figure()
    plt.plot(rewards_per_min)
    if show:
        plt.show()

def get_trial_end_types(task_entry):
    hdf_filename = get_hdf_file(task_entry)
    hdf = tables.openFile(hdf_filename)
    task_msgs = hdf.root.task_msgs[:]

    # number of successful trials
    reward_msgs = filter(lambda m: m[0] == 'reward', task_msgs)
    n_success_trials = len(reward_msgs)
    
    # number of hold errors
    hold_penalty_inds = np.array(filter(lambda k: task_msgs[k][0] == 'hold_penalty', range(len(task_msgs))))
    msg_before_hold_penalty = task_msgs[(hold_penalty_inds - 1).tolist()]
    n_terminus_hold_errors = len(filter(lambda m: m['msg'] == 'terminus_hold', msg_before_hold_penalty))
    n_origin_hold_errors = len(filter(lambda m: m['msg'] == 'origin_hold', msg_before_hold_penalty))

    # number of timeout trials
    timeout_msgs = filter(lambda m: m[0] == 'timeout_penalty', task_msgs)
    n_timeout_trials = len(timeout_msgs)

def get_hold_error_rate(task_entry):
    hold_error_rate = float(n_terminus_hold_errors) / n_success_trials
    return hold_error_rate

def get_center_out_reach_inds(hdf, fixed=True):
    task_msgs = hdf.root.task_msgs[:]

    if fixed:
        update_bmi_msgs = np.nonzero(task_msgs['msg'] == 'update_bmi')[0]
        if len(update_bmi_msgs) > 0:
            fixed_start = update_bmi_msgs[-1] + 1
        else:
            fixed_start = 0
        task_msgs = task_msgs[fixed_start:]

    n_msgs = len(task_msgs)
    terminus_hold_msg_inds = np.array(filter(lambda k: task_msgs[k]['msg'] == 'terminus_hold', range(n_msgs)))
    terminus_msg_inds = terminus_hold_msg_inds - 1

    boundaries = np.vstack([task_msgs[terminus_msg_inds]['time'], 
                            task_msgs[terminus_hold_msg_inds]['time']]).T
    return boundaries

def get_movement_durations(task_entry):
    '''
    Get the movement durations of each trial which enters the 'terminus_hold'
    state
    '''
    hdf = get_hdf(task_entry)
    boundaries = get_center_out_reach_inds(hdf)

    return np.diff(boundaries, axis=1) * 1./60
    

def lookup_task_entry(task_entry):
    '''
    Enable multiple ways to specify a task entry, e.g. by primary key or by
    object
    '''
    if isinstance(task_entry, models.TaskEntry):
        pass
    elif isinstance(task_entry, int):
        task_entry = get_task_entry(task_entry)
    return task_entry

def get_reach_trajectories(task_entry, rotate=True):
    task_entry = lookup_task_entry(task_entry)
    hdf = get_hdf(task_entry)
    boundaries = get_center_out_reach_inds(hdf)
    targets = hdf.root.task[:]['target']
    cursor = hdf.root.task[:]['cursor']

    n_trials = boundaries.shape[0]
    trajectories = [None] * n_trials
    for k, (st, end) in enumerate(boundaries):
        trial_target = targets[st][[0,2]]
        angle = -np.arctan2(trial_target[1], trial_target[0])

        # counter-rotate trajectory
        cursor_pos_tr = cursor[st:end, [0,2]]
        trial_len = cursor_pos_tr.shape[0]
        if rotate:
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        else:
            R = np.eye(2)
        trajectories[k] = np.dot(R, cursor_pos_tr.T)
    return trajectories

def get_movement_error(task_entry):
    '''
    Get movement error
    '''
    task_entry = lookup_task_entry(task_entry)
    reach_trajectories = get_reach_trajectories(task_entry)

    n_trials = len(reach_trajectories)

    print reach_trajectories[0].shape
    ME = np.array([np.mean(np.abs(x[1, ::6])) for x in reach_trajectories])
    MV = np.array([np.std(np.abs(x[1, ::6])) for x in reach_trajectories])

    return ME, MV

def plot_trajectories(task_entry, ax=None, show=False, **kwargs):
    hdf = get_hdf(task_entry)
    boundaries = get_center_out_reach_inds(hdf)
    targets = hdf.root.task[:]['target']
    cursor = hdf.root.task[:]['cursor']

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    n_trials = boundaries.shape[0]
    for k, (st, end) in enumerate(boundaries):
        trial_target = targets[st][[0,2]]
        angle = -np.arctan2(trial_target[1], trial_target[0])

        # counter-rotate trajectory
        cursor_pos_tr = cursor[st:end, [0,2]]
        trial_len = cursor_pos_tr.shape[0]
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        cursor_pos_tr_rot = np.vstack([np.dot(R, cursor_pos_tr[k,:]) for k in range(trial_len)])
        ax.plot(cursor_pos_tr_rot[:,0], cursor_pos_tr_rot[:,1], **kwargs)

    if show:
        plt.show()

def get_workspace_size(task_entry):
    '''
    Get movement error
    '''
    hdf = get_hdf(task_entry)
    targets = hdf.root.task[:]['target']
    print targets.min(axis=0)
    print targets.max(axis=0)

def plot_dist_to_targ(task_entry, targ_dist=10., plot_all=False, ax=None, **kwargs):
    task_entry = lookup_task_entry(task_entry)
    reach_trajectories = get_reach_trajectories(task_entry)
    target = np.array([targ_dist, 0])
    from utils.geometry import l2norm
    trajectories_dist_to_targ = [l2norm(traj.T - target, axis=0) for traj in reach_trajectories]

    trajectories_dist_to_targ = map(lambda x: x[::6], trajectories_dist_to_targ)
    max_len = np.max([len(traj) for traj in trajectories_dist_to_targ])
    n_trials = len(trajectories_dist_to_targ)

    # TODO use masked arrays
    data = np.ones([n_trials, max_len]) * np.nan
    for k, traj in enumerate(trajectories_dist_to_targ):
        data[k, :len(traj)] = traj

    mean_dist_to_targ = np.array([nanmean(data[:,k]) for k in range(max_len)])

    if ax == None:
        plt.figure()
        ax = plt.subplot(111)

    # time vector, assuming original screen update rate of 60 Hz
    time = np.arange(max_len)*0.1
    if plot_all:
        for dist_to_targ in trajectories_dist_to_targ: 
            ax.plot(dist_to_targ, **kwargs)
    else:
        ax.plot(time, mean_dist_to_targ, **kwargs)

    import plot
    plot.set_ylim(ax, [0, targ_dist])
    plot.ylabel(ax, 'Distance to target')
    plt.draw()

def get_task_entries_by_date(date, subj=None):
    '''
    Get all the task entries for a particular date
    '''
    kwargs = dict(date__year=date.year, date__month=date.month,
                  date__day=date.day)
    if isinstance(subj, str) or isinstance(subj, unicode):
        kwargs['subject__name__startswith'] = str(subj)
    elif subj is not None:
        kwargs['subject__name'] = subj.name
    return list(models.TaskEntry.objects.filter(**kwargs))
