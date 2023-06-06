'''
Interface between the Django database methods/models and data analysis code
'''
# django initialization
import os
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
from .boot_django import boot_django
from django.db import connection
from django.db.utils import OperationalError

import sys
import json
import numpy as np
import datetime
import pickle
import tables
from collections import defaultdict

from .tracker import models

# configure django
boot_django()

# decorator for mysql access
def mysqlreconnect(f):
    def inner(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except OperationalError:
            connection.close()
            return f(*args, **kwargs)
    return inner

class TaskEntry(object):
    '''
    Wrapper class for the TaskEntry django class so that object-oriented methods
    can be defined for TaskEntry blocks (e.g., for analysis methods for a particular experiment)
    without needing to modfiy the database model.
    '''
    def __init__(self, task_entry_id, dbname='default', **kwargs):
        '''
        Constructor for TaskEntry

        Parameters
        ----------
        task_entry_id : int
            'id' number associated with the TaskEntry (database primary key)
        dbname : string, optional, default='default'
            Name of database (if multiple databases are available). Database must
            be declared in db/settings.py
        **kwargs : dict
            For any additional parameters unused by child classes

        Returns
        -------
        TaskEntry instance
        '''
        self.dbname = dbname
        if isinstance(task_entry_id, models.TaskEntry):
            self.record = task_entry_id
        else:
            self.record = models.TaskEntry.objects.using(self.dbname).get(id=task_entry_id)
        self.id = self.record.id
        self.params = self.record.params
        if (isinstance(self.params, str) or isinstance(self.params, str)) and len(self.params) > 0:
            self.params = json.loads(self.record.params)

            # Add the params dict to the object's dict
            for key, value in list(self.params.items()):
                try:
                    setattr(self, key, value)
                except AttributeError: # if the object already has an attribute of the same name ...
                    setattr(self, key + '_param', value)
                except:
                    import traceback
                    traceback.print_exc()

                if isinstance(value, list):
                    self.params[key] = tuple(value)

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
        try:
            self.report = json.loads(self.record.report)
        except:
            self.report = ''


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
            decoder_objects = [x.load() for x in records]
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
            print(key, report[key])

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
            filt = kwargs.pop('trial_filter_fn', default_trial_filter_fn)
        if cond == None:
            cond = kwargs.pop('trial_condition_fn', default_trial_condition_fn)
        if proc == None:
            proc = kwargs.pop('trial_proc_fn', default_trial_proc_fn)
        if comb == None:
            comb = kwargs.pop('data_comb_fn', default_data_comb_fn)


        trial_filter_fn = filt
        trial_condition_fn = cond
        trial_proc_fn = proc
        data_comb_fn = comb

        # import trial_filter_functions, trial_proc_functions, trial_condition_functions
        # if isinstance(trial_filter_fn, str):
        #     trial_filter_fn = getattr(trial_filter_functions, trial_filter_fn)

        # if isinstance(trial_proc_fn, str):
        #     trial_proc_fn = getattr(trial_proc_functions, trial_proc_fn)

        # if isinstance(trial_condition_fn, str):
        #     trial_condition_fn = getattr(trial_condition_functions, trial_condition_fn)

        te = self
        trial_msgs = [msgs for msgs in te.trial_msgs if trial_filter_fn(te, msgs)]
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

        if len(list(newdata.keys())) == 1:
            key = list(newdata.keys())[0]
            return newdata[key]
        else:
            return newdata

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
        try:
            task_supplement = models.System.objects.get(name='task_supplement')
            supplementary_data_file = os.path.join(task_supplement.path, '%d.mat' % self.record.id)
        except:
            import warnings
            warnings.warn('No "task_supplement" system. Please add one.')
            supplementary_data_file = None
        if supplementary_data_file is None:
            data = fn()
            setattr(self, '_%s' % key, data)
            return getattr(self, '_%s' % key)
        if not os.path.exists(supplementary_data_file):
            data = fn()
            savemat(supplementary_data_file, dict(key=data))
            setattr(self, '_%s' % key, data)
            return getattr(self, '_%s' % key)
        else:
            supplementary_data = loadmat(supplementary_data_file)
            if (not key in supplementary_data) or clean:
                supplementary_data[key] = fn()
                savemat(supplementary_data_file, supplementary_data)
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
        q = models.DataFile.objects.using(self.record._state.db).filter(entry_id=self.id, system__name='hdf')
        if len(q) == 0:
            # empty string for HDF filename if none is linked in the database
            return ''
        elif len(q) == 1:
            q = q[0]
            # dbconfig = getattr(config, 'db_config_%s' % self.record._state.db)
            # data_path = models.KeyValueStore.get('data_path', default='/storage', dbname=self.record._state.db)
            return os.path.join(q.system.path, q.path)

    @property
    def hdf(self):
        '''
        Return a reference to the HDF file recorded during this TaskEntry
        '''
        if not hasattr(self, 'hdf_file'):
            self.hdf_file = tables.open_file(self.hdf_filename, mode='r')
        return self.hdf_file

    def close_hdf(self):
        if hasattr(self, 'hdf_file'):
            self.hdf_file.close()
            delattr(self, 'hdf_file')

    def close(self):
        '''
        Cleanup functions
        '''
        self.close_hdf()

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

            # parse out events
            from riglib.dio import parse
            self.strobe_data = parse.parse_data(self._plx.events[:].data)
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
            self._decoder_obj = self.decoder_record.load(encoding='latin1')
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

    def get_datafile(self, system_name, **query_kwargs):
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
        file_names = [os.path.join(q.system.path, q.path) for q in file_records]
        if len(file_names) == 1:
            return file_names[0]
        else:
            return list(file_names)

    @property
    def plx_filename(self):
        '''
        Return the name of the plx file associated with this TaskEntry
        '''
        return self.get_datafile('plexon')

    @property
    def plx2_filename(self):
        '''
        Returns the name of any files associated with the plexon2 system. Used briefly only in a rig where there were multiple recording systems used simultaneously.
        '''
        return self.get_datafile('plexon2')

    @property
    def bmiparams_filename(self):
        '''
        Return the name of the npz file that was used to store CLDA parameters, if one exists
        '''
        return self.get_datafile('bmi_params')

    @property
    def blackrock_filenames(self):
        x = self.get_datafile(system_name='blackrock')
        y = self.get_datafile(system_name='blackrock2')

        if type(x) is not list:
            x = [x]
        if type(y) is not list:
            y = [y]

        return x+y
    @property
    def nev_filename(self):
        '''
        Get any blackrock nev files associated with this TaskEntry
        '''
        return self.get_datafile(system_name='blackrock', path__endswith='.nev')

    @property
    def nsx_filenames(self):
        '''
        Get any blackrock nsx files associated with this TaskEntry
        '''
        return self.get_datafile(system_name='blackrock', path__endswith='.nsx')

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
        '''
        Save the current task entry (and associated records) to a different database
        '''
        current_db = self.record._state.db
        if current_db == new_db:
            print("new database and current database are the same!")
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

    def get_state_inds(self, state_name):
        '''
        Determine the rows of the task table in which the task is in a particular state.
        '''
        task_msgs = self.hdf.root.task_msgs[:]
        n_msgs = len(task_msgs)
        msg_inds = np.array([k for k in range(n_msgs) if task_msgs[k]['msg'] == state_name])

        inds = np.zeros(len(self.hdf.root.task), dtype=bool)
        for idx in msg_inds:
            if task_msgs[idx+1]['msg'] == 'None':
                break
            for k in range(task_msgs[idx]['time'], task_msgs[idx+1]['time']+1):
                inds[k] = 1
        return inds



class TaskEntryCollection(object):
    '''
    Container for analyzing multiple task entries with an arbitrarily deep hierarchical structure
    '''
    def __init__(self, blocks, name='', cls=TaskEntry, **kwargs):
        '''
        Constructor for TaskEntryCollection

        Parameters
        ----------
        blocks : np.iterable
            Some iterable object which contains TaskEntry ID numbers to look up in the database
        name : string, optional, default=''
            Name to give this collection
        cls : type, optional, default=TaskEntry
            class constructor to use for all the IDs. Uses the generic TaskEntry defined above by default

        Returns
        -------
        '''
        self.block_ids = blocks
        self.blocks = parse_blocks(blocks, cls=cls, **kwargs)
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
            filt = kwargs.pop('trial_filter_fn', default_trial_filter_fn)
        if cond == None:
            cond = kwargs.pop('trial_condition_fn', default_trial_condition_fn)
        if proc == None:
            proc = kwargs.pop('trial_proc_fn', default_trial_proc_fn)
        if comb == None:
            comb = kwargs.pop('data_comb_fn', default_data_comb_fn)


        trial_filter_fn = filt
        trial_condition_fn = cond
        trial_proc_fn = proc
        data_comb_fn = comb

        # import trial_filter_functions, trial_proc_functions, trial_condition_functions
        # if isinstance(trial_filter_fn, str):
        #     trial_filter_fn = getattr(trial_filter_functions, trial_filter_fn)

        # if isinstance(trial_proc_fn, str):
        #     trial_proc_fn = getattr(trial_proc_functions, trial_proc_fn)

        # if isinstance(trial_condition_fn, str):
        #     trial_condition_fn = getattr(trial_condition_functions, trial_condition_fn)


        result = []
        error_count = 0
        for blockset in self.blocks:
            if not np.iterable(blockset):
                blockset = (blockset,)

            blockset_data = defaultdict(list)
            for te in blockset:
                if verbose:
                    print(".")

                # Filter out the trials you want
                trial_msgs = [msgs for msgs in te.trial_msgs if trial_filter_fn(te, msgs)]
                n_trials = len(trial_msgs)

                ## Call a function on each trial
                for k in range(n_trials):
                    try:
                        output = trial_proc_fn(te, trial_msgs[k], **kwargs)
                        trial_condition = trial_condition_fn(te, trial_msgs[k])
                        blockset_data[trial_condition].append(output)
                    except:
                        error_count += 1
                        print(trial_msgs[k])
                        import traceback
                        traceback.print_exc()
                        if error_count > max_errors:
                            raise Exception

            # Aggregate the data from the blockset, which may include multiple task entries
            blockset_data_comb = dict()
            for key in blockset_data:
                blockset_data_comb[key] = data_comb_fn(blockset_data[key])
            if len(list(blockset_data_comb.keys())) == 1:
                key = list(blockset_data_comb.keys())[0]
                result.append(blockset_data_comb[key])
            else:
                result.append(blockset_data_comb)

        if verbose:
            sys.stdout.write('\n')
        return result

    def proc_blocks(self, filt=None, proc=None, cond=None, comb=None, verbose=False, return_type=list, **kwargs):
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
        if filt == None:
            filt = kwargs.pop('block_filter_fn', default_block_filter_fn)
        if cond == None:
            cond = kwargs.pop('block_condition_fn', default_trial_condition_fn)
        if proc == None:
            proc = kwargs.pop('block_proc_fn', default_trial_proc_fn)
        if comb == None:
            comb = kwargs.pop('data_comb_fn', default_data_comb_fn)


        block_filter_fn = filt
        block_condition_fn = cond
        block_proc_fn = proc
        data_comb_fn = comb

        # Look up functions by name, if strings are given instead of functions
        # import trial_filter_functions, trial_proc_functions, trial_condition_functions
        # if isinstance(block_filter_fn, str):
        #     block_filter_fn = getattr(trial_filter_functions, block_filter_fn)

        # if isinstance(block_proc_fn, str):
        #     block_proc_fn = getattr(block_proc_functions, block_proc_fn)

        result = []
        for blockset in self.blocks:
            if not np.iterable(blockset):
                blockset = (blockset,)

            blockset_data = []
            for te in blockset:
                if verbose:
                    print(".")

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


##################
# Query functions
##################
@mysqlreconnect
def get_task_entries_by_id(ids):
    '''
    Returns list of task entries according to the given id or multiple ids
    '''
    return list(models.TaskEntry.objects.filter(pk__in=ids))

@mysqlreconnect
def get_task_entries_by_date(startdate, enddate=datetime.date.today(), dbname='default'):
    '''
    Returns list of task entries within date range (inclusive). startdate and enddate
    are date objects. End date is optional- today's date by default.
    '''
    return list(models.TaskEntry.objects.using(dbname).filter(date__gte=startdate).filter(date__lte=enddate))

@mysqlreconnect
def get_task_entries(subj=None, date=None, task=None, dbname='default', **kwargs):
    '''
    Get all the task entries for a particular date

    Parameters
    ----------
    subj: string, optional, default=None
        Specify the beginning of the name of the subject or the full name. If not specified, blocks from all subjects are returned
    date: multiple types, optional, default=today
        Query date for blocks. The date can be specified as a datetime.date object, a tuple of (start, end) datetime.date objects,
        or a 3-tuple of ints (year, month, day).
    task: string
        name of task to filter
    kwargs: dict, optional
        Additional keyword arguments to pass to models.TaskEntry.objects.filter
    '''

    if isinstance(date, datetime.date):
        kwargs.update(dict(date__year=date.year, date__month=date.month, date__day=date.day))
    elif isinstance(date, tuple) and len(date) == 2:
        kwargs.update(dict(date__gte=date[0], date__lte=date[1]))
    elif isinstance(date, tuple) and len(date) == 3:
        kwargs.update(dict(date__year=date[0], date__month=date[1], date__day=date[2]))

    if isinstance(subj, str):
        kwargs['subject__name__startswith'] = str(subj)
    elif subj is not None:
        kwargs['subject__name'] = subj.name
    
    if isinstance(task, str):
        kwargs['task__name__startswith'] = task
    elif task is not None:
        kwargs['task__name'] = task.name

    return list(models.TaskEntry.objects.using(dbname).filter(**kwargs))

#################
# Collections
###############
def parse_blocks(blocks, cls=TaskEntry, **kwargs):
    '''
    Parse out a hierarchical structure of block ids. Used to construct TaskEntryCollection objects
    '''
    data = []
    for block in blocks:
        if np.iterable(block):
            te = parse_blocks(block, cls=cls, **kwargs)
        else:
            te = cls(block, **kwargs)
        data.append(te)
    return data

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

    keys = list(keyed_ids.keys())
    keys.sort()

    grouped_ids = []
    for date in keys:
        grouped_ids.append(tuple(keyed_ids[date]))
    return grouped_ids

############
# Statistics
############
def default_data_comb_fn(x):
    return x

def default_trial_filter_fn(te, trial_msgs):
    return True

def default_block_filter_fn(te):
    return True

def default_trial_proc_fn(te, trial_msgs):
    return 1

def default_trial_condition_fn(te, trial_msgs):
    return 0

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

def lookup_task_entries(*task_entry, dbname='default'):
    '''
    Enable multiple ways to specify a task entry, e.g. by primary key or by
    object
    '''
    if len(task_entry) == 1:
        task_entry = task_entry[0]
        if isinstance(task_entry, models.TaskEntry):
            pass
        elif isinstance(task_entry, int):
            task_entry = models.TaskEntry.objects.using(dbname).get(pk=task_entry) #get_task_entry(task_entry)
        return task_entry
    else:
        return [lookup_task_entries(x) for x in task_entry]

def get_task_id(name, dbname='default'):
    '''
    Returns the task ID for the specified task name.
    '''
    return models.Task.objects.using(dbname).get(name=name).pk

def get_decoder_entry(entry, dbname='default'):
    '''Returns the database entry for the decoder used in the session. Argument can be a task entry
    or the ID number of the decoder entry itself.
    '''
    if isinstance(entry, int):
        return models.Decoder.objects.using(dbname).get(pk=entry)
    else:
        params = json.loads(entry.params)
        if 'decoder' in params:
            return models.Decoder.objects.using(dbname).get(pk=params['decoder'])
        elif 'bmi' in params:
            return models.Decoder.objects.using(dbname).get(pk=params['bmi'])
        else:
            return None

def get_decoder_name(entry, dbname='default'):
    '''
    Returns the filename of the decoder used in the session.
    Takes TaskEntry object.
    '''
    entry = lookup_task_entries(entry)
    try:
        decid = json.loads(entry.params)['decoder']
    except:
        decid = json.loads(entry.params)['bmi']
    return models.Decoder.objects.using(dbname).get(pk=decid).path

def get_decoder_name_full(entry, dbname='default'):
    entry = lookup_task_entries(entry)
    decoder_basename = get_decoder_name(entry)
    sys_path = models.System.objects.using(dbname).get(name='bmi')
    return os.path.join(sys_path, decoder_basename)

def get_decoder(entry, dbname='default'):
    entry = lookup_task_entries(entry)
    filename = get_decoder_name_full(entry)
    dec = pickle.load(open(filename, 'r'))
    dec.db_entry = get_decoder_entry(entry, dbname=dbname)
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

def get_task_name(entry, dbname='default'):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return models.Task.objects.using(dbname).get(pk=entry.task_id).name

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

def get_subject(entry, dbname='default'):
    '''
    Returns name of subject for session.
    Takes TaskEntry object.
    '''
    return models.Subject.objects.using(dbname).get(pk=entry.subject_id).name

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
    print("Subject: ", get_subject(entry))
    print("Task: ", get_task_name(entry))
    print("Date: ", str(get_date(entry)))
    hours = np.floor(get_length(entry)/3600)
    mins = np.floor(get_length(entry)/60) - hours*60
    secs = get_length(entry) - mins*60
    print("Length: " + str(int(hours))+ ":" + str(int(mins)) + ":" + str(int(secs)))
    try:
        print("Assist level: ", get_param(entry,'assist_level'))
    except:
        print("Assist level: 0")
    print("Completed trials: ", get_completed_trials(entry))
    print("Success rate: ", get_success_rate(entry)*100, "%")
    print("Reward rate: ", get_reward_rate(entry), "trials/minute")
    print("Initiate rate: ", get_initiate_rate(entry), "trials/minute")

###############
# Decoder query
###############
def get_decoder_parent(decoder, dbname='default'):
    '''
    decoder = database record of decoder object
    '''
    entryid = decoder.entry_id
    te = get_task_entries_by_id(entryid, dbname=dbname)[0]
    return get_decoder_entry(te)

def get_decoder_sequence(decoder, dbname='default'):
    '''
    decoder = database record of decoder object
    '''
    parent = get_decoder_parent(decoder, dbname=dbname)
    if parent is None:
        return [decoder]
    else:
        return [decoder] + get_decoder_sequence(parent, dbname=dbname)

def search_by_decoder(decoder, dbname='default'):
    '''
    Returns task entries that used specified decoder. Decoder argument can be
    decoder entry or entry ID.
    '''

    if isinstance(decoder, int):
        decid = decoder
    else:
        decid = decoder.id
    blocks = list(models.TaskEntry.objects.using(dbname).filter(params__contains='"bmi": '+str(decid))) + list(models.TaskEntry.objects.using(dbname).filter(params__contains='"decoder": '+str(decid)))
    return blocks

def search_by_units(unitlist, decoderlist=None, exact=False, dbname='default'):
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
        all_decoders = models.Decoder.objects.using(dbname).all()
    subset = set(tuple(unit) for unit in unitlist)
    dec_list = []
    for dec in all_decoders:
        try:
            decobj = dec.get()
            decset = set(tuple(unit) for unit in decobj.units)
            if subset==decset:
                dec_list = dec_list + [dec]
            elif not exact and subset.issubset(decset):
                dec_list = dec_list + [dec]
        except:
            pass
    return dec_list

def load_last_decoder():
    '''
    Returns the decoder object corresponding to the last decoder trained and
    added to the database
    '''
    all_decoder_records = models.Decoder.objects.all()
    record = all_decoder_records[len(all_decoder_records)-1]
    return record.load()


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
    return [x for x in models.TaskEntry.objects.filter(**kwargs) if x.id > id]

def get_bmi_blocks(date, subj='C'):
    blocks = TaskEntrySet.get_blocks(filter_fns=[min_trials(50)], subj=subj, date=date, task__name__startswith='bmi') + TaskEntrySet.get_blocks(filter_fns=[min_trials(5)], subj=subj, date=date, task__name__startswith='clda')
    blocks.sort(key=lambda x: x.date)
    return blocks

def bmi_filt(te):
    bmi_feat = models.Feature.objects.using(te._state.db).get(name='bmi')
    return bmi_feat in te.feats.all()

##################
## aolab functions
##################
def get_entry_details(entry):
    '''
    Returns subject, id, and date for the given entry
    '''
    subject = get_subject(entry)
    te = entry.id
    date = get_date(entry).date()
    return subject, te, date

def get_sessions(subject, date, task_name, session_name=None, exclude_ids=[], filter_fn=lambda x:True, dbname='default'):
    '''
    Returns list of subject, date, and id for all sessions on the given date
    '''
    if session_name:
        entries = get_task_entries(subj=subject, task=task_name, date=date, session=session_name, dbname=dbname)
    else:
        entries = get_task_entries(subj=subject, task=task_name, date=date, dbname=dbname)
    if len(entries) == 0:
        import warnings
        warnings.warn("No entries found")
        return [], [], []
    subject, te, date = zip(*[get_entry_details(e) for e in entries if filter_fn(e) and e.id not in exclude_ids])
    return subject, te, date
    
def get_mc_sessions(subject, date, mc_task_name='manual control', dbname='default', **kwargs):
    '''
    Returns list of subject, date, and id for all bmi control sessions on the given date
    '''
    return get_sessions(subject, date, task_name=mc_task_name, dbname=dbname, **kwargs)
    
def get_bmi_sessions(subject, date, session_name='training', bmi_task_name='bmi control', dbname='default', **kwargs):
    '''
    Returns list of subject, date, and id for all bmi control sessions on the given date
    '''
    return get_sessions(subject, date, session_name=session_name, task_name=bmi_task_name, dbname=dbname, **kwargs)


def get_preprocessed_sources(entry):
    '''
    Returns a list of datasource names that should be preprocessed for the given entry
    '''
    params = entry.task_params
    sources = ['exp']
    if 'record_headstage' in params and params['record_headstage']:
        sources.append('broadband')
        sources.append('lfp')
    sources.append('eye')
    
    return sources
    

def get_rawfiles_for_taskentry(e, system_subfolders=None):
    '''
    Gets the hdf and ecube files associated with each task entry 
    
    Args:
        entry (TaskEntry): recording to find raw files for 
        system_subfolders (dict, optional): dictionary of system subfolders where the 
           data for that system is located. If None, defaults to the system name
    
    Returns: 
        files : list of (system, filepath) for each datafile associated with this task entry 
    '''
    file_list = e.get_data_files(dbname=e._state.db)
    files = {}
    for df in file_list:
        if df.system.name == 'optitrack' or df.system.name == 'e3v':
            path = os.path.normpath(df.path)
            parts = path.split(os.sep)
            if len(parts) > 1:
                filepath = os.path.join(*parts[1:])
            filepath = filepath.replace(':', '_')
        else:
            filepath = os.path.basename(df.path)
        if system_subfolders is None:
            files[df.system.name] = os.path.join(df.system.name, filepath)
        elif df.system.name in system_subfolders:
            files[df.system.name] = os.path.join(system_subfolders[df.system.name], filepath)
        else:
            files[df.system.name] = filepath

    return files
