'''
Interface between the Django database methods/models and data analysis code
'''
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import sys
import json
import numpy as np
import datetime
import pickle
import cPickle
import tables
import matplotlib.pyplot as plt
import time, datetime
from scipy.stats import nanmean
import db
from collections import defaultdict, OrderedDict
from analysis import trial_filter_functions, trial_proc_functions, trial_condition_functions

from config import config

try:
    import plotutil
except:
    pass

from tracker import models

# default DB, change this variable from python session to switch to other database
db_name = 'default'

def group_ids(ids, grouping_fn=lambda te: te.calendar_date):
    '''
    Automatically group together a flat list of database IDs

    Parameters
    ----------
    ids: iterable
        iterable of ints representing the ID numbers of TaskEntry objects to group
    grouping_fn: callable, optional (default=sort by date); call signature: grouping_fn(task_entry)
        Takes a dbfn.TaskEntry as its only argument and returns a hashable and sortable object
        by which to group the ids
    '''
    keyed_ids = defaultdict(list)
    for id in ids:
        te = TaskEntry(id)
        key = grouping_fn(te)
        keyed_ids[key].append(id)

    keys = keyed_ids.keys()
    keys.sort()

    grouped_ids = []
    for date in keys:
        grouped_ids.append(tuple(keyed_ids[date]))
    return grouped_ids

def default_data_comb_fn(x):
    return x

def get_records_of_trained_decoders(task_entry):
    '''
    Returns unpickled decoder objects that were trained in a specified session.
    '''
    task_entry = lookup_task_entries(task_entry)
    records = models.Decoder.objects.filter(entry_id=task_entry.id)
    records = list(records)
    if len(records) == 1:
        return records[0]
    else:
        return records

def lookup_task_entries(*task_entry):
    '''
    Enable multiple ways to specify a task entry, e.g. by primary key or by
    object
    '''
    if len(task_entry) == 1:
        task_entry = task_entry[0]
        if isinstance(task_entry, models.TaskEntry):
            pass
        elif isinstance(task_entry, int):
            task_entry = models.TaskEntry.objects.using(db_name).get(pk=task_entry) #get_task_entry(task_entry)
        return task_entry
    else:
        return [lookup_task_entries(x) for x in task_entry]

def get_task_id(name):
    '''
    Returns the task ID for the specified task name.
    '''
    return models.Task.objects.using(db_name).get(name=name).pk

def get_decoder_entry(entry):
    '''Returns the database entry for the decoder used in the session. Argument can be a task entry
    or the ID number of the decoder entry itself.
    '''
    if isinstance(entry, int):
        return models.Decoder.objects.using(db_name).get(pk=entry)
    else:
        params = json.loads(entry.params)
        if 'decoder' in params:
            return models.Decoder.objects.using(db_name).get(pk=params['decoder'])
        elif 'bmi' in params:
            return models.Decoder.objects.using(db_name).get(pk=params['bmi'])
        else:
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
    return models.Decoder.objects.using(db_name).get(pk=decid).path

def get_decoder_name_full(entry):
    entry = lookup_task_entries(entry)
    decoder_basename = get_decoder_name(entry)
    return os.path.join(db.paths.pathdict[dbname], 'decoders', decoder_basename)

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

def get_param(entry,paramname):
    '''
    Returns parameter value.
    Takes TaskEntry object.
    '''
    return json.loads(entry.params)[paramname]

def get_task_name(entry):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return models.Task.objects.using(db_name).get(pk=entry.task_id).name
    
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
    return models.Subject.objects.using(db_name).get(pk=entry.subject_id).name
    
def get_length(entry):
    '''
    Returns length of session in seconds.
    Takes TaskEntry object.
    '''
    try:
        report = json.loads(entry.report)
    except:
        return 0.0
    return report[-1][2]-report[0][2]
    
def get_success_rate(entry):
    '''
    Returns (# of trials rewarded)/(# of trials intiated).
    Takes TaskEntry object.
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    total=0.0
    rew=0.0
    for s in report:
        if s[0]=='reward':
            rew+=1
            total+=1
        if s[0]=='hold_penalty' or s[0]=='timeout_penalty':
            total+=1
    return rew/total

def get_completed_trials(entry):
    '''
    Returns # of trials rewarded
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    return len([s for s in report if s[0]=="reward"])

def get_initiate_rate(entry):
    '''
    Returns average # of trials initated per minute.
    Takes TaskEntry object.
    '''
    length = get_length(entry)
    try:
        report = json.loads(entry.report)
    except: return 0.0
    count=0.0
    for s in report:
        if s[0]=='reward' or s[0]=='hold_penalty' or s[0]=='timeout_penalty':
            count+=1
    return count/(length/60.0)

def get_reward_rate(entry):
    '''
    Returns average # of trials completed per minute.
    Takes TaskEntry object.
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    count=0.0
    rewardtimes = []
    for s in report:
        if s[0]=='reward':
            count+=1
            rewardtimes.append(s[2])
    if len(rewardtimes)==0:
        return 0
    else:
        length = rewardtimes[-1] - report[0][2]
        return count/(length/60.0)
    
def session_summary(entry):
    '''
    Prints a summary of info about session.
    Takes TaskEntry object.
    '''
    entry = lookup_task_entries(entry)
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
    return models.TaskEntry.objects.using(db_name).filter(date__gte=startdate).filter(date__lte=enddate)
    
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

def search_by_decoder(decoder):
    '''
    Returns task entries that used specified decoder. Decoder argument can be
    decoder entry or entry ID.
    '''

    if isinstance(decoder, int):
        decid = decoder
    else:
        decid = decoder.id
    blocks = list(models.TaskEntry.objects.using(db_name).filter(params__contains='"bmi": '+str(decid))) + list(models.TaskEntry.objects.using(db_name).filter(params__contains='"decoder": '+str(decid))) 
    return blocks

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
        all_decoders = models.Decoder.objects.using(db_name).all()
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

def get_task_entries_by_date(subj=None, date=datetime.date.today(), dbname='default', **kwargs):
    '''
    Get all the task entries for a particular date

    Parameters
    ----------
    subj: string, optional, default=None
        Specify the beginning of the name of the subject or the full name. If not specified, blocks from all subjects are returned
    date: multiple types, optional, default=today
        Query date for blocks. The date can be specified as a datetime.date object 
        or a 3-tuple (year, month, day). 
    kwargs: dict, optional
        Additional keyword arguments to pass to models.TaskEntry.objects.filter
    '''
    
    if isinstance(date, datetime.date):
        kwargs.update(dict(date__year=date.year, date__month=date.month, date__day=date.day))
    elif isinstance(date, tuple) and len(date) == 3:
        kwargs.update(dict(date__year=date[0], date__month=date[1], date__day=date[2]))
    else:
        raise ValueError("Cannot interpret date: %r" % date)

    if isinstance(subj, str) or isinstance(subj, unicode):
        kwargs['subject__name__startswith'] = str(subj)
    elif subj is not None:
        kwargs['subject__name'] = subj.name

    return list(models.TaskEntry.objects.using(dbname).filter(**kwargs))

def load_last_decoder():
    '''
    Returns the decoder object corresponding to the last decoder trained and
    added to the database
    '''
    all_decoder_records = models.Decoder.objects.all()
    record = all_decoder_records[len(all_decoder_records)-1]
    return record.load()



class TaskEntry(object):
    '''
    Wrapper class for the TaskEntry django class
    '''
    def __init__(self, task_entry_id, dbname='default', **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.dbname = dbname
        if isinstance(task_entry_id, models.TaskEntry):
            self.record = task_entry_id
        else:
            self.record = models.TaskEntry.objects.using(self.dbname).get(id=task_entry_id)
        self.id = self.record.id
        self.params = self.record.params
        if (isinstance(self.params, str) or isinstance(self.params, unicode)) and len(self.params) > 0:
            self.params = json.loads(self.record.params)

            # Add the params dict to the object's dict
            for key, value in self.params.items():
                try:
                    setattr(self, key, value)
                except AttributeError:
                    setattr(self, key + '_param', value)
                except:
                    pass

        self.date = self.record.date

        ## Extract a date month-day-year date for determining if other blocks were on the same day
        self.calendar_date = datetime.datetime(self.date.year, self.date.month, self.date.day)

        self.notes = self.record.notes
        self.subject = models.Subject.objects.using(dbname).get(pk=self.record.subject_id).name

        # Load decoder record
        if 'decoder' in self.params:
            self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(pk=self.params['decoder'])
        elif 'bmi' in self.params:
            self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(pk=self.params['bmi'])
        else: # Try direct lookup
            try:
                self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(entry_id=self.id)
            except:
                self.decoder_record = None

        # Load the event log (report)
        self.report = json.loads(self.record.report)

    def get_decoders_trained_in_block(self, return_type='record'):
        '''
        Retrieve decoders associated with this block. A block may have multiple decoders 
        associated with it, e.g., a single decoder seeding block may be used to generate several seed decoders

        Parameters
        ----------
        return_type: string
            'record' means the Django database records are returned. 'object' means the un-pickled riglib.bmi.Decoder objects are returned

        Returns
        -------
        list or object
            If only one decoder is linked to this task entry, an object is returned (either a db record or a Decoder instance). If multiple decoders are linked, 
            a list of objects is returned
        '''
        records = models.Decoder.objects.using(self.dbname).filter(entry_id=self.id)
        if return_type == 'record':
            decoder_objects = list(records)
        elif return_type == 'object':
            decoder_objects = map(lambda x: x.load(), records)
        else:
            raise ValueError("Unrecognized return_type!")
        if len(decoder_objects) == 1: decoder_objects = decoder_objects[0]
        return decoder_objects

    def summary_stats(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        report = self.record.offline_report()
        for key in report:
            print key, report[key]

    def proc(self, filt=None, proc=None, cond=None, comb=None, **kwargs):
        '''
        Generic trial-level data analysis function

        Parameters
        ----------
        filt: callable; call signature: trial_filter_fn(trial_msgs)
            Function must return True/False values to determine if a set of trial messages constitutes a valid set for the analysis
        proc: callable; call signature: trial_proc_fn(task_entry, trial_msgs)
            The main workhorse function 
        cond: callable; call signature: trial_condition_fn(task_entry, trial_msgs)
            Determine what the trial *subtype* is (useful for separating out various types of catch trials)
        comb: callable; call signature: data_comb_fn(list)
            Combine the list into the desired output structure
        kwargs: optional keyword arguments
            For 'legacy' compatibility, you can also specify 'trial_filter_fn' for 'filt', 'trial_proc_fn' for 'proc', 
            'trial_condition_fn' for 'cond', and 'data_comb_fn' for comb. These are ignored if any newer equivalents are specified.
            All other keyword arguments are passed to the 'proc' function. 

        Returns
        -------
        result: list
            The results of all the analysis. The length of the returned list equals len(self.blocks). Sub-blocks
            grouped by tuples are combined into a single result. 

        '''
        if filt == None:
            filt = kwargs.pop('trial_filter_fn', trial_filter_functions.default)
        if cond == None:
            cond = kwargs.pop('trial_condition_fn', trial_condition_functions.default)
        if proc == None:
            proc = kwargs.pop('trial_proc_fn', trial_proc_functions.default)
        if comb == None:
            comb = kwargs.pop('data_comb_fn', default_data_comb_fn)


        trial_filter_fn = filt
        trial_condition_fn = cond
        trial_proc_fn = proc
        data_comb_fn = comb


        if isinstance(trial_filter_fn, str):
            trial_filter_fn = getattr(trial_filter_functions, trial_filter_fn)

        if isinstance(trial_proc_fn, str):
            trial_proc_fn = getattr(trial_proc_functions, trial_proc_fn)            

        if isinstance(trial_condition_fn, str):
            trial_condition_fn = getattr(trial_condition_functions, trial_condition_fn)


        te = self
        trial_msgs = filter(lambda msgs: trial_filter_fn(te, msgs), te.trial_msgs)
        n_trials = len(trial_msgs)
        
        blockset_data = defaultdict(list)
        
        ## Call a function on each trial    
        for k in range(n_trials):
            output = trial_proc_fn(te, trial_msgs[k], **kwargs)
            trial_condition = trial_condition_fn(te, trial_msgs[k])
            blockset_data[trial_condition].append(output)
        
        newdata = dict()
        for key in blockset_data:
            newdata[key] = data_comb_fn(blockset_data[key])

        if len(newdata.keys()) == 1:
            key = newdata.keys()[0]
            return newdata[key]
        else:
            return newdata

    @property 
    def supplementary_data_file(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return '/storage/task_supplement/%d.mat' % self.record.id

    def get_cached_attr(self, key, fn, clean=False):
        '''
        Generic method for saving the results of a computation to the supplementary_data_file associated with this block

        Parameters
        ----------
        key : string
            Variable name in cache file 
        fn : callable
            Function to execute (no arguments) to get the required data, if it is not present in the file
        clean : bool, default=False
            If true, force the recomputation of the data product even if it is already present in the cache file
        '''
        from scipy.io import savemat, loadmat
        if hasattr(self, '_%s' % key):
            return getattr(self, '_%s' % key)
        if not os.path.exists(self.supplementary_data_file):
            data = fn()
            savemat(self.supplementary_data_file, dict(key=data))
            setattr(self, '_%s' % key, data)
            return getattr(self, '_%s' % key)
        else:
            supplementary_data = loadmat(self.supplementary_data_file)
            if (not key in supplementary_data) or clean:
                supplementary_data[key] = fn()
                savemat(self.supplementary_data_file, supplementary_data)
            setattr(self, '_%s' % key, supplementary_data[key])
            return getattr(self, '_%s' % key)            

    def get_matching_state_transition_seq(self, seq):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        task_msgs = self.hdf.root.task_msgs[:]
        seq = np.array(seq, dtype='|S256')
        msg_list_inds = []
        trial_msgs = []
        epochs = []
        for k in range(len(task_msgs)-len(seq)):
            if np.all(task_msgs[k:k+len(seq)]['msg'] == seq):
                msg_list_inds.append(k)
                trial_msgs.append(task_msgs[k:k+len(seq)])  
                epochs.append((task_msgs[k]['time'], task_msgs[k+len(seq)-1]['time']))      
        epochs = np.vstack(epochs)
        return msg_list_inds, trial_msgs, epochs

    def get_task_var_during_epochs(self, epochs, var_name, comb_fn=lambda x:x, start_offset=0):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        data = []
        for st, end in epochs:
            st += start_offset
            epoch_data = self.hdf.root.task[st:end][var_name]

            # Determine whether all the rows in the extracted sub-table are the same
            # This code is stolen from the interweb:
            # http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
            a = epoch_data
            b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
            unique_epoch_data = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

            # unique_epoch_data = np.unique(epoch_data.ravel())
            if len(unique_epoch_data) == 1:
                data.append(unique_epoch_data)
            else:
                data.append(epoch_data)
        return comb_fn(data)

    @property
    def hdf_filename(self):
        '''
        Get the task-generated HDF file linked to this TaskEntry
        '''
        q = models.DataFile.objects.using(self.record._state.db).get(entry_id=self.id, system__name='hdf')
        dbconfig = getattr(config, 'db_config_%s' % self.record._state.db)
        # dbconfig = getattr(config, 'db_config_exorig')
        # dbconfig = getattr(config, 'db_config_bmi3d')
        return os.path.join(dbconfig['data_path'], 'rawdata', q.system.name, q.path)

    @property
    def hdf(self):
        '''
        Return a reference to the HDF file recorded during this TaskEntry
        '''
        if not hasattr(self, 'hdf_file'):
            try:
                self.hdf_file = tables.open_file(self.hdf_filename)
            except:
                self.hdf_file = tables.openFile(self.hdf_filename)
        return self.hdf_file

    @property
    def plx(self):
        '''
        Return a reference to the opened plx file recorded during this TaskEntry
        '''
        try:
            self._plx
        except:
            from plexon import plexfile
            self._plx = plexfile.openFile(str(self.plx_filename))
        return self._plx

    @property
    def task(self):
        '''
        Return database record of the task performed during this TaskEntry
        '''
        return self.record.task

    @property
    def decoder(self):
        '''
        Return a reference to the unpickled riglib.bmi.Decoder instance associated with this TaskEntry.
        This function will error if there is no decoder actually associated with this TaskEntry
        '''
        if not hasattr(self, '_decoder_obj'):
            self._decoder_obj = self.decoder_record.load()
        return self._decoder_obj

    @property
    def clda_param_hist(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        bmi_params = models.System.objects.get(name='bmi_params')
        q = models.DataFile.objects.filter(entry_id=self.id, system_id=bmi_params.id)

        if not hasattr(self, '_clda_param_hist'):
            if hasattr(self.hdf.root, 'clda'):
                self._clda_param_hist = self.hdf.root.clda[:]
            elif len(q) > 0: #get_bmiparams_file(self.record) is not None:
                fname = os.path.join(q[0].system.path, q[0].path)
                self._clda_param_hist = np.load(fname)
            else:
                self._clda_param_hist = None
        return self._clda_param_hist

    @property
    def length(self):
        '''
        Return the length of the TaskEntry, in seconds. Only works for tasks which inherit from LogExperiment, as it uses the event log for determining timing.
        '''
        try:
            report = json.loads(self.record.report)
        except:
            return 0.0
        return report[-1][2]-report[0][2]

    def get_datafile(self, system_name, intermediate_path='rawdata', **query_kwargs):
        '''
        Look up the file linked to this TaskEntry from a specific system

        Parameters
        ----------
        system_name : string
            Name of system (must match the database record name)
        intermediate_path: string
            specific directory structure for this system. Some systems have inconsistent directory structures
        **query_kwargs: keyword arguments
            These are passed to the Django record 'filter' function

        Returns
        -------
        filename of associated file 
        '''
        file_records = models.DataFile.objects.using(self.record._state.db).filter(entry_id=self.id, system__name=system_name, **query_kwargs)
        dbconfig = getattr(config, 'db_config_%s' % self.record._state.db)
        file_names = [os.path.join(dbconfig['data_path'], intermediate_path, os.path.basename(q.system.path), q.path) for q in file_records]
        if len(file_names) == 1:
            return file_names[0]
        else:
            return list(file_names)

    @property
    def plx_filename(self):
        '''
        Return the name of the plx file associated with this TaskEntry
        '''
        return self.get_datafile('plexon', intermediate_path='plexon')

    @property
    def plx2_filename(self):
        '''
        Returns the name of any files associated with the plexon2 system. Used briefly only in a rig where there were multiple recording systems used simultaneously.
        '''
        return self.get_datafile('plexon2', intermediate_path='')

    @property 
    def bmiparams_filename(self):
        '''
        Return the name of the npz file that was used to store CLDA parameters, if one exists
        '''
        return self.get_datafile('bmi_params', intermediate_path='rawdata')

    @property 
    def blackrock_filenames(self):
        return self.get_datafile(system_name='blackrock', intermediate_path='')

    @property
    def nev_filename(self):
        '''
        Get any blackrock nev files associated with this TaskEntry
        '''
        return self.get_datafile(system_name='blackrock', intermediate_path='', path__endswith='.nev')

    @property
    def nsx_filenames(self):
        '''
        Get any blackrock nsx files associated with this TaskEntry
        '''
        return self.get_datafile(system_name='blackrock', intermediate_path='', path__endswith='.nsx')

    @property
    def decoder_filename(self):
        '''
        Get filename of riglib.bmi.Decoder object associate with this TaskEntry. Only works for BMI tasks!
        '''
        return self.decoder_record.filename #get_datafile('bmi', intermediate_path='rawdata')
        # decoder_basename = self.decoder_record.path
        # dbconfig = getattr(config, 'db_config_%s' % self.record._state.db)
        # return os.path.join(dbconfig['data_path'], 'decoders', decoder_basename)

    @property
    def name(self):
        '''
        Return the 'name' of the TaskEntry used to set all the names of linked data files
        '''
        # TODO this needs to be hacked because the current way of determining a 
        # a filename depends on the number of things in the database, i.e. if 
        # after the fact a record is removed, the number might change. read from
        # the file instead
        return str(os.path.basename(self.hdf_filename).rstrip('.hdf'))

    def trained_decoder_filenames(self):
        decoder_records = self.get_decoders_trained_in_block()
        filenames = []
        if np.iterable(decoder_records):
            for rec in decoder_records:
                filenames.append(rec.filename)
        else:
            filenames.append(decoder_records.filename)
        return filenames

    def __str__(self):
        return self.record.__str__()

    def __repr__(self):
        return self.record.__repr__()

    @property
    def task_type(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if 'bmi' in self.task.name: 
            return 'BMI'
        elif 'clda' in self.task.name:
            return 'CLDA'

    @property
    def n_rewards(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        # of rewards given during a block. This number could be different
        from the total number of trials if children of this class over-ride
        the n_trials calculator to exclude trials of a certain type, e.g. BMI
        trials in which the subject was assiste
        '''
        return self.record.offline_report()['Total rewards']

    @property
    def total_reward_time(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.n_rewards * self.reward_time

    @property 
    def datafiles(self):
        '''
        Returns a dictionary where keys are system names and values are the location(s) of associated data files
        '''
        datafiles = models.DataFile.objects.using(self.record._state.db).filter(entry_id=self.id)
        files = dict((d.system.name, d.get_path()) for d in datafiles)    
        return files

    def save_to_database(self, new_db):
        current_db = self.record._state.db
        if current_db == new_db:
            print "new database and current database are the same!"
            return

        # save the subject
        self.record.subject.save(using=new_db)

        # save the task
        self.record.task.save(using=new_db)        

        # save the task entry record
        self.record.save(using=new_db)

        # save the records of the datafiles associated with the task entry
        # A bit of circuswork is required to move the DataFiles because they have a foreign key for the system that correspomds to the datafile
        datafiles = models.DataFile.objects.using(current_db).filter(entry_id=self.id)
        for d in datafiles:
            sys_name = d.system.name
            d.save(using=new_db)
            new_sys = models.System.objects.using(new_db).get(name=d.system.name)
            d.system = new_sys
            d.save()            

        # save any decoder records used by this block
        if not (self.decoder_record is None):
            self.decoder_record.save(using=new_db)


class TaskEntrySet(object):
    def __init__(self, blocks, name=''):
        from analysis import performance
        self.task_entries = map(performance._get_te, blocks)
        self.name = name

    def map(self, fn):
        return np.array(map(fn, self.task_entries))

    def __getattr__(self, attr):
        return self.map(lambda te: getattr(te, attr))

    def boxplot(self, attr, ax, plotattr=lambda x: '', xlabel=True):
        ax.boxplot(getattr(self, attr))
        if xlabel:
            ticklabels = [te.plot_ticklabel + plotattr(te) for te in self.task_entries]
            ax.set_xticklabels(ticklabels)

    def scatterplot(self, attr, ax, plotattr=lambda x: '', xlabel=True):
        data = getattr(self, attr)
        plotutil.set_axlim(ax, [0, len(data)])
        plot_pts = np.arange(0., len(data)) + 0.5
        ax.scatter(plot_pts, data)
        ax.set_xticklabels(plot_pts)
        ax.set_xticks(plot_pts)

        if xlabel:
            ticklabels = [te.plot_ticklabel + plotattr(te) for te in self.task_entries]
            ax.set_xticklabels(ticklabels)
        ax.set_xlim([0, len(data)])

    def histogram(self, fn, ax, bins, labels=None):
        if labels == None:
            labels = ['']*len(self.task_entries)
        colors = plotutil.colors.values()
        for k, te in enumerate(self.task_entries):
            data = fn(te)
            plotutil.histogram_line(ax, data, bins, color=colors[k], linewidth=2, label=labels[k])
            
    def init_plot(self, ax):
        if ax == None:
            plt.figure()
            ax = plt.subplot(111)
        return ax
        
    def plot_reach_time(self, ax=None, **kwargs):
        ax = self.init_plot(ax)
        self.boxplot('reach_time', ax, **kwargs)
        plotutil.set_axlim(ax, [0, 4], [0, 1, 2, 3, 4], axis='y')
        plotutil.ylabel(ax, 'Reach time (s)')

    def plot_ME(self, ax=None, **kwargs):
        ax = self.init_plot(ax)
        self.boxplot('ME', ax, **kwargs)
        plotutil.set_axlim(ax, [0, 2.], axis='y')
        plotutil.ylabel(ax, 'Movement\nerror (cm)')
        
    def plot_MV(self, ax=None, **kwargs):
        ax = self.init_plot(ax)
        self.boxplot('MV', ax, **kwargs)
        plotutil.set_axlim(ax, [0, 1.], axis='y')
        plotutil.ylabel(ax, 'Movement\nvariability (cm)')
        
    def plot_perc_correct(self, ax=None, **kwargs):
        ax = self.init_plot(ax)
        self.scatterplot('perc_correct', ax, **kwargs)
        plotutil.set_axlim(ax, [0.6, 1.], labels=[0.6, 0.7, 0.8, 0.9, 1.0], axis='y')
        plotutil.ylabel(ax, '% correct', offset=-0.06)

    def plot_perf_summary(self, **kwargs):
        plt.figure(figsize=(8,8), facecolor='w')
        axes = plotutil.subplots(4, 1, return_flat=True, y=0.05)
        self.plot_reach_time(axes[0], xlabel=False, **kwargs)
        self.plot_ME(axes[1], xlabel=False, **kwargs)
        self.plot_MV(axes[2], xlabel=False, **kwargs)
        self.plot_perc_correct(axes[3], xlabel=True, **kwargs)
        plt.suptitle(self.name)


    @classmethod
    def construct_from_queryset(cls, name='', filter_fns=[], **kwargs):
        blocks = get_task_entries_by_date(**kwargs)
        if name == '': name = str(kwargs)

        # iteratively apply the filter functions
        for fn in filter_fns:
            blocks = filter(fn, blocks)

        task_entry_set = TaskEntrySet(blocks, name=name)
        return task_entry_set
        
    @classmethod
    def get_blocks(cls, name='', filter_fns=[], **kwargs):
        blocks = get_task_entries_by_date(**kwargs)
        if name == '': name = str(kwargs)

        # iteratively apply the filter functions
        for fn in filter_fns:
            blocks = filter(fn, blocks)

        return blocks



def parse_blocks(blocks, **kwargs):
    '''
    Parse out a hierarchical structure of block ids. Used to construct TaskEntryCollection objects
    '''
    data = []
    from analysis import performance
    for block in blocks:
        if np.iterable(block):
            te = parse_blocks(block, **kwargs)
        else:
            te = performance._get_te(block, **kwargs)
        data.append(te)
    return data


class TaskEntryCollection(object):
    '''
    Container for analyzing multiple task entries with an arbitrarily deep hierarchical structure
    '''
    def __init__(self, blocks, name='', **kwargs):
        '''
        Constructor for TaskEntryCollection

        Parameters
        ----------
        blocks: np.iterable
            Some iterable object which contains TaskEntry ID numbers to look up in the database
        name: string, optional, default=''
            Name to give this collection 

        Returns
        -------
        '''
        self.block_ids = blocks
        self.blocks = parse_blocks(blocks, **kwargs)
        self.kwargs = kwargs
        self.name = name

    def __len__(self):
        return len(self.blocks)

    def proc_trials(self, filt=None, proc=None, cond=None, comb=None, verbose=False, max_errors=10, **kwargs):
        '''
        Generic framework to perform a trial-level analysis on the entire dataset

        Parameters
        ----------
        filt: callable; call signature: trial_filter_fn(trial_msgs)
            Function must return True/False values to determine if a set of trial messages constitutes a valid set for the analysis
        proc: callable; call signature: trial_proc_fn(task_entry, trial_msgs)
            The main workhorse function 
        cond: callable; call signature: trial_condition_fn(task_entry, trial_msgs)
            Determine what the trial *subtype* is (useful for separating out various types of catch trials)
        comb: callable; call signature: data_comb_fn(list)
            Combine the list into the desired output structure
        verbose: boolean, optional, default = True
            Feedback print statements so that you know processing is happening
        max_errors: int, optional, default = 10
            Number of trials resulting in error before the processing quits. Below this threshold, errors are printed but the code continues on to the next trial.

        Returns
        -------
        result: list
            The results of all the analysis. The length of the returned list equals len(self.blocks). Sub-blocks
            grouped by tuples are combined into a single result. 
        '''

        if filt == None:
            filt = kwargs.pop('trial_filter_fn', trial_filter_functions.default)
        if cond == None:
            cond = kwargs.pop('trial_condition_fn', trial_condition_functions.default)
        if proc == None:
            proc = kwargs.pop('trial_proc_fn', trial_proc_functions.default)
        if comb == None:
            comb = kwargs.pop('data_comb_fn', default_data_comb_fn)


        trial_filter_fn = filt
        trial_condition_fn = cond
        trial_proc_fn = proc
        data_comb_fn = comb


        if isinstance(trial_filter_fn, str):
            trial_filter_fn = getattr(trial_filter_functions, trial_filter_fn)

        if isinstance(trial_proc_fn, str):
            trial_proc_fn = getattr(trial_proc_functions, trial_proc_fn)            

        if isinstance(trial_condition_fn, str):
            trial_condition_fn = getattr(trial_condition_functions, trial_condition_fn)


        result = []
        error_count = 0
        for blockset in self.blocks:
            if not np.iterable(blockset):
                blockset = (blockset,)
            
            blockset_data = defaultdict(list)
            for te in blockset:
                if verbose:
                    print "."
          
                # Filter out the trials you want
                trial_msgs = filter(lambda msgs: trial_filter_fn(te, msgs), te.trial_msgs)
                n_trials = len(trial_msgs)
        
                ## Call a function on each trial    
                for k in range(n_trials):
                    try:
                        output = trial_proc_fn(te, trial_msgs[k], **kwargs)
                        trial_condition = trial_condition_fn(te, trial_msgs[k])
                        blockset_data[trial_condition].append(output)
                    except:
                        error_count += 1
                        print trial_msgs[k]
                        import traceback
                        traceback.print_exc()
                        if error_count > max_errors:
                            raise Exception
        
            # Aggregate the data from the blockset, which may include multiple task entries
            blockset_data_comb = dict()
            for key in blockset_data:
                blockset_data_comb[key] = data_comb_fn(blockset_data[key])
            if len(blockset_data_comb.keys()) == 1:
                key = blockset_data_comb.keys()[0]
                result.append(blockset_data_comb[key])
            else:
                result.append(blockset_data_comb)

        if verbose:
            sys.stdout.write('\n')
        return result

    def proc_blocks(self, block_filter_fn=trial_filter_functions.default, block_proc_fn=trial_proc_functions.default, 
                    data_comb_fn=default_data_comb_fn, verbose=True, return_type=list, **kwargs):
        '''
        Generic framework to perform a block-level analysis on the entire dataset, 
        e.g., percent of trials correct, which require analyses across trials

        Parameters
        ----------
        block_filter_fn: callable; call signature: block_filter_fn(task_entry)
            Function must return True/False values to determine if a task entry is valid for the analysis
        block_proc_fn: callable; call signature: trial_proc_fn(task_entry)
            The main workhorse function 
        data_comb_fn: callable; call signature: data_comb_fn(list)
            Combine the list into the desired output structure

        Returns
        -------
        result: list
            The results of all the analysis. The length of the returned list equals len(self.blocks). Sub-blocks
            grouped by tuples are combined into a single result. 
        '''
        # Look up functions by name, if strings are given instead of functions
        if isinstance(block_filter_fn, str):
            block_filter_fn = getattr(trial_filter_functions, block_filter_fn)

        if isinstance(block_proc_fn, str):
            block_proc_fn = getattr(block_proc_functions, block_proc_fn)

        result = []
        for blockset in self.blocks:
            if not np.iterable(blockset):
                blockset = (blockset,)
            
            blockset_data = []
            for te in blockset:
                if verbose:
                    print "."

                if block_filter_fn(te):
                    blockset_data.append(block_proc_fn(te, **kwargs))

            blockset_data = data_comb_fn(blockset_data)
            result.append(blockset_data)

        if verbose:
            sys.stdout.write('\n')
        return return_type(result)

    def __repr__(self):
        if not self.name == '':
            return str(self.block_ids)
        else:
            return "TaskEntryCollection: ", self.name

    def __str__(self):
        return self.__repr__()

######################
## Filter functions
######################
def _assist_level_0(task_entry_model):
    te = TaskEntry(task_entry_model)
    if 'assist_level' in te.params:
        assist_level = te.params['assist_level']
        return np.all(np.array(assist_level) == 0)
    else:
        return True

def using_decoder(decoder_type):
    from tasks import performance
    def _check_decoder(task_entry_model):
        te = performance._get_te(task_entry_model)
        try:
            return (decoder_type in str(te.decoder_type))
        except:
            return False
    return _check_decoder

def min_trials(min_trial_count):
    from tasks import performance
    def fn(task_entry_model):
        te = performance._get_te(task_entry_model)
        try:
            return te.n_trials >= min_trial_count
        except:
            return False
    return fn

##########
## Filters
##########
def get_blocks_after(id, **kwargs):
    return filter(lambda x: x.id > id, models.TaskEntry.objects.filter(**kwargs))

def get_bmi_blocks(date, subj='C'):
    blocks = TaskEntrySet.get_blocks(filter_fns=[min_trials(50)], subj=subj, date=date, task__name__startswith='bmi') + TaskEntrySet.get_blocks(filter_fns=[min_trials(5)], subj=subj, date=date, task__name__startswith='clda')
    blocks.sort(key=lambda x: x.date)
    return blocks

#############################################################
## Deprecated code
#############################################################

def deprecation_warning():
    import warnings
    warnings.warn("dbfunctions: this function is now deprecated and eventually will be removed. Use TaskEntry instead!!!!!!")

def get_decoders_trained_in_block(task_entry, dbname='default'):
    '''
    Returns unpickled decoder objects that were trained in a specified session.
    '''
    deprecation_warning()
    te = TaskEntry(task_entry, db_name)
    return te.get_decoders_trained_in_block(return_type='object')

def get_decoder_ids_trained_in_block(task_entry):
    '''
    Returns the database entries of decoders trained in a particular session.
    '''
    deprecation_warning()
    te = TaskEntry(task_entry, db_name)
    return te.get_decoders_trained_in_block(return_type='record')

def get_hdf_file(entry):
    '''
    Returns the name of the hdf file associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.hdf_filename

def get_hdf(entry):
    '''
    Return hdf opened file
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.hdf

def get_plx_file(entry):
    '''
    Returns the name of the plx file associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.plx_filename
        
def get_plx2_file(entry):
    '''
    Returns the name of the plx2 file associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.plx2_filename
        
def get_bmiparams_file(entry):
    '''
    Returns the name of the bmi parameter update history file associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.bmiparams_filename

def get_blackrock_files(entry):
    '''
    Returns a list containing the names of the blackrock files (there could be more
    than one) associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.blackrock_filenames

def get_nev_file(entry):
    '''
    Returns the name of the nev file associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.nev_filename    

def get_nsx_files(entry):
    '''
    Returns a list containing the names of the nsx files (there could be more
    than one) associated with the session.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.nsx_filenames

def get_decoder_name(entry):
    ''' 
    Returns the filename of the decoder used in the session.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.decoder_record.name

def get_decoder_name_full(entry):
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.decoder_filename  

def get_decoder(entry):
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.decoder

def get_params(entry):
    '''
    Returns a dict of all task params for session.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.params

def get_param(entry, paramname):
    '''
    Returns parameter value.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.params[paramname]

def get_task_name(entry):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return models.Task.objects.using(db_name).get(pk=entry.task_id).name
    
def get_date(entry):
    '''
    Returns date and time of session (as a datetime object).
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.date    
    
def get_notes(entry):
    '''
    Returns notes for session.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.notes
    
def get_subject(entry):
    '''
    Returns name of subject for session.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.subject
    
def get_length(entry):
    '''
    Returns length of session in seconds.
    Takes TaskEntry object.
    '''
    deprecation_warning()
    te = TaskEntry(entry, db_name)
    return te.length

def get_binned_spikes_file(entry):
    ''' Return binned spike file if it exists'''
    entry = lookup_task_entries(entry)
    fname = db.paths.data_path+'binned_spikes/'+entry.name+'.npz'
    print fname
    if os.path.isfile(fname):
        return np.load(fname)
    else:
        print 'Not found'
        return None
    
def get_success_rate(entry):
    '''
    Returns (# of trials rewarded)/(# of trials intiated).
    Takes TaskEntry object.
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    total=0.0
    rew=0.0
    for s in report:
        if s[0]=='reward':
            rew+=1
            total+=1
        if s[0]=='hold_penalty' or s[0]=='timeout_penalty':
            total+=1
    return rew/total

def get_completed_trials(entry):
    '''
    Returns # of trials rewarded
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    return len([s for s in report if s[0]=="reward"])

def get_initiate_rate(entry):
    '''
    Returns average # of trials initated per minute.
    Takes TaskEntry object.
    '''
    length = get_length(entry)
    try:
        report = json.loads(entry.report)
    except: return 0.0
    count=0.0
    for s in report:
        if s[0]=='reward' or s[0]=='hold_penalty' or s[0]=='timeout_penalty':
            count+=1
    return count/(length/60.0)

def get_reward_rate(entry):
    '''
    Returns average # of trials completed per minute.
    Takes TaskEntry object.
    '''
    try:
        report = json.loads(entry.report)
    except: return 0.0
    count=0.0
    rewardtimes = []
    for s in report:
        if s[0]=='reward':
            count+=1
            rewardtimes.append(s[2])
    if len(rewardtimes)==0:
        return 0
    else:
        length = rewardtimes[-1] - report[0][2]
        return count/(length/60.0)
    
def session_summary(entry):
    '''
    Prints a summary of info about session.
    Takes TaskEntry object.
    '''
    entry = lookup_task_entries(entry)
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
   
