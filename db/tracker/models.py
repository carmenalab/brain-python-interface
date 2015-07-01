'''
Django database modules. See https://docs.djangoproject.com/en/dev/intro/tutorial01/
for a basic introduction
'''

import os
import json
import cPickle, pickle
import inspect
from collections import OrderedDict
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

import numpy as np

from riglib import calibrations, experiment
from config import config

def _get_trait_default(trait):
    '''Function which tries to resolve traits' retarded default value system'''
    _, default = trait.default_value()
    if isinstance(default, tuple) and len(default) > 0:
        try:
            func, args, _ = default
            default = func(*args)
        except:
            pass
    return default

class Task(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(default=True, blank=True)
    def __unicode__(self):
        return self.name
    
    def get(self, feats=()):
        from namelist import tasks
        from riglib import experiment
        if self.name in tasks:
            return experiment.make(tasks[self.name], Feature.getall(feats))
        else:
            return experiment.Experiment

    @staticmethod
    def populate():
        from namelist import tasks
        real = set(tasks.keys())
        db = set(task.name for task in Task.objects.all())
        for name in real - db:
            Task(name=name).save()

    def params(self, feats=(), values=None):
        from riglib import experiment
        from namelist import instance_to_model
        import plantlist
        if values is None:
            values = dict()
        
        params = OrderedDict()
        Exp = self.get(feats=feats)
        ctraits = Exp.class_traits()

        def add_trait(trait):
            varname = dict()
            varname['type'] = ctraits[trait].trait_type.__class__.__name__
            varname['default'] = _get_trait_default(ctraits[trait])
            varname['desc'] = ctraits[trait].desc
            varname['hidden'] = 'hidden' if Exp.is_hidden(trait) else 'visible'
            if trait in values:
                varname['value'] = values[trait]
            if varname['type'] == "Instance":
                Model = instance_to_model[ctraits[trait].trait_type.klass]
                insts = Model.objects.order_by("-date")#[:200]
                varname['options'] = [(i.pk, i.name) for i in insts]
            if varname['type'] == "Enum":
                varname['options'] = getattr(Exp, trait + '_options')
            params[trait] = varname
            if trait == 'bmi':
                params['decoder'] = varname

        ordered_traits = Exp.ordered_traits
        for trait in ordered_traits:
            if trait in Exp.class_editable_traits():
                add_trait(trait)

        for trait in Exp.class_editable_traits():
            if trait not in params and not Exp.is_hidden(trait):
                add_trait(trait)

        for trait in Exp.class_editable_traits():
            if trait not in params:
                add_trait(trait)

        return params

    def sequences(self):
        from json_param import Parameters
        seqs = dict()
        for s in Sequence.objects.filter(task=self.id):
            seqs[s.id] = s.to_json()

        return seqs

class Feature(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(blank=True, default=True)
    def __unicode__(self):
        return self.name

    @property
    def desc(self):
        return self.get().__doc__
    
    def get(self):
        from namelist import features
        return features[self.name]

    @staticmethod
    def populate():
        from namelist import features
        real = set(features.keys())
        db = set(feat.name for feat in Feature.objects.all())
        for name in real - db:
            Feature(name=name).save()

    @staticmethod
    def getall(feats):
        features = []
        for feat in feats:
            if isinstance(feat, (int, float, str, unicode)):
                try:
                    feat = Feature.objects.get(pk=int(feat)).get()
                except ValueError:
                    try:
                        feat = Feature.objects.get(name=feat).get()
                    except:
                        print "Cannot find feature %s"%feat
                        continue
            elif isinstance(feat, models.Model):
                feat = feat.get()
            
            features.append(feat)
        return features

class System(models.Model):
    name = models.CharField(max_length=128)
    path = models.TextField()
    archive = models.TextField()

    def __unicode__(self):
        return self.name
    
    @staticmethod
    def populate():
        for name in ["eyetracker", "hdf", "plexon", "bmi", "bmi_params", "juice_log", "blackrock"]:
            try:
                System.objects.get(name=name)
            except ObjectDoesNotExist:
                System(name=name, path="/storage/rawdata/%s"%name).save()

class Subject(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Generator(models.Model):
    name = models.CharField(max_length=128)
    params = models.TextField()
    static = models.BooleanField()
    visible = models.BooleanField(blank=True, default=True)

    def __unicode__(self):
        return self.name
    
    def get(self):
        '''
        Retrieve the function that can be used to construct the ..... generator? sequence?
        '''
        from namelist import generators
        return generators[self.name]

    @staticmethod
    def populate():
        from namelist import generators
        listed_generators = set(generators.keys())
        db_generators = set(gen.name for gen in Generator.objects.all())

        # determine which generators are missing from the database using set subtraction
        missing_generators = listed_generators - db_generators
        for name in missing_generators:
            # The sequence/generator constructor can either be a callable or a class constructor... not aware of any uses of the class constructor
            try:
                args = inspect.getargspec(generators[name]).args
                print args
            except TypeError:
                args = inspect.getargspec(generators[name].__init__).args
                args.remove("self")
            
            # A generator is determined to be static only if it takes an "exp" argument representing the Experiment class
            static = ~("exp" in args)
            if "exp" in args:
                args.remove("exp")

            # TODO not sure why the 'length' argument is being removed; is it assumed that all generators will take a 'length' argument?
            if "length" in args:
                args.remove("length")

            gen_obj = Generator(name=name, params=",".join(args), static=static)
            gen_obj.save()

    def to_json(self, values=None):
        if values is None:
            values = dict()
        gen = self.get()
        try:
            args = inspect.getargspec(gen)
            names, defaults = args.args, args.defaults
        except TypeError:
            args = inspect.getargspec(gen.__init__)
            names, defaults = args.args, args.defaults
            names.remove("self")

        # if self.static:
        #     defaults = (None,)+defaults
        # else:
        #     #first argument is the experiment
        #     names.remove("exp")
        # arginfo = zip(names, defaults)

        params = dict()
        from itertools import izip
        for name, default in izip(names, defaults):
            if name == 'exp':
                continue
            typename = "String"

            params[name] = dict(type=typename, default=default, desc='')
            if name in values:
                params[name]['value'] = values[name]

        return dict(name=self.name, params=params)

class Sequence(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    generator = models.ForeignKey(Generator)
    name = models.CharField(max_length=128)
    params = models.TextField() #json data
    sequence = models.TextField(blank=True) #pickle data
    task = models.ForeignKey(Task)

    def __unicode__(self):
        return self.name
    
    def get(self):
        from riglib.experiment import generate
        from json_param import Parameters

        if self.generator.static:
            ## If the generator is static, 
            if len(self.sequence) > 0:
                return generate.runseq, dict(seq=cPickle.loads(str(self.sequence)))
            else:
                return generate.runseq, dict(seq=self.generator.get()(**Parameters(self.params).params))            
        else:
            return self.generator.get(), Parameters(self.params).params

    def to_json(self):
        from json_param import Parameters
        state = 'saved' if self.pk is not None else "new"
        js = dict(name=self.name, state=state)
        js['static'] = len(self.sequence) > 0
        js['params'] = self.generator.to_json(Parameters(self.params).params)['params']
        js['generator'] = self.generator.id, self.generator.name
        return js

    @classmethod
    def from_json(cls, js):
        '''
        Construct a models.Sequence instance from JSON data (e.g., generated by the web interface for starting experiments)
        '''
        from json_param import Parameters

        # Error handling when input argument 'js' actually specifies the primary key of a Sequence object already in the database
        try:
            return Sequence.objects.get(pk=int(js))
        except:
            pass
        
        # Make sure 'js' is a python dictionary
        if not isinstance(js, dict):
            js = json.loads(js)

        # Determine the ID of the "generator" used to make this sequence
        genid = js['generator']
        if isinstance(genid, (tuple, list)):
            genid = genid[0]
        
        # Construct the database record for the new Sequence object
        seq = cls(generator_id=int(genid), name=js['name'])

        # Link the generator instantiation parameters to the sequence record
        # Parameters are stored in JSON format in the database
        seq.params = Parameters.from_html(js['params']).to_json()

        # If the sequence is to be static, 
        if js['static']:
            print "db.tracker.models.Sequence.from_json: storing static sequence data to database"
            generator_params = Parameters(seq.params).params
            seq_data = seq.generator.get()(**generator_params)
            seq.sequence = cPickle.dumps(seq_data)
        return seq

class TaskEntry(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    task = models.ForeignKey(Task)
    feats = models.ManyToManyField(Feature)
    sequence = models.ForeignKey(Sequence)

    params = models.TextField()
    report = models.TextField()
    notes = models.TextField()
    visible = models.BooleanField(blank=True, default=True)
    backup = models.BooleanField(blank=True, default=False)

    def __unicode__(self):
        return "{date}: {subj} on {task} task, id={id}".format(
            date=self.date.strftime("%h. %e, %Y, %l:%M %p"),
            subj=self.subject.name,
            task=self.task.name,
            id=self.id)
    
    def get(self, feats=()):
        from json_param import Parameters
        from riglib import experiment
        Exp = experiment.make(self.task.get(), tuple(f.get() for f in self.feats.all())+feats)
        params = Parameters(self.params)
        params.trait_norm(Exp.class_traits())
        if issubclass(Exp, experiment.Sequence):
            gen, gp = self.sequence.get()
            seq = gen(Exp, **gp)
            exp = Exp(seq, **params.params)
        else:
            exp = Exp(**params.params)
        exp.event_log = json.loads(self.report)
        return exp
    
    @property
    def task_params(self):
        from json_param import Parameters
        data = Parameters(self.params).params
        if 'bmi' in data:
            data['decoder'] = data['bmi']
        ##    del data['bmi']
        return data

    def plexfile(self, path='/storage/plexon/', search=False):
        rplex = Feature.objects.get(name='relay_plexon')
        rplexb = Feature.objects.get(name='relay_plexbyte')
        feats = self.feats.all()
        if rplex not in feats and rplexb not in feats:
            return None

        if not search:
            system = System.objects.get(name='plexon')
            df = DataFile.objects.filter(entry=self.id, system=system)
            if len(df) > 0:
                return df[0].get_path()
        
        if len(self.report) > 0:
            event_log = json.loads(self.report)
            import os, sys, glob, time
            if len(event_log) < 1:
                return None

            start = event_log[-1][2]
            files = sorted(glob.glob(path+"/*.plx"), key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < 60:
                     return files[0]

    def offline_report(self):
        Exp = self.task.get(self.feats.all())
        task = self.task.get(self.feats.all())
        report = json.loads(self.report)
        rpt = Exp.offline_report(report)

        ## If this is a BMI block, add the decoder name to the report (doesn't show up properly in drop-down menu for old blocks)
        try:
            from db import dbfunctions
            te = dbfunctions.TaskEntry(self.id, dbname=self._state.db)
            rpt['Decoder name'] = te.decoder_record.name
        except AttributeError:
            pass
        except:
            import traceback
            traceback.print_exc()
        return rpt

    def to_json(self):
        '''
        Create a JSON dictionary of the metadata associated with this block for display in the web interface
        '''
        from json_param import Parameters

        # Run the metaclass constructor for the experiment used. If this can be avoided it would help to break some of the cross-package software dependencies,
        # making it easier to analyze data without installing software for the entire rig
        Exp = self.task.get(self.feats.all())

        state = 'completed' if self.pk is not None else "new"
        js = dict(task=self.task.id, state=state, subject=self.subject.id, notes=self.notes)
        js['feats'] = dict([(f.id, f.name) for f in self.feats.all()])
        js['params'] = self.task.params(self.feats.all(), values=self.task_params)

        # Supply sequence generators which are declared to be compatible with the selected task class
        exp_generators = dict() 
        if hasattr(Exp, 'sequence_generators'):
            for seqgen_name in Exp.sequence_generators:
                try:
                    g = Generator.objects.get(name=seqgen_name)
                    exp_generators[g.id] = seqgen_name
                except:
                    pass
        js['generators'] = exp_generators


        ## Add data files to the web interface. To be removed (never ever used)
        if issubclass(self.task.get(), experiment.Sequence):
            js['sequence'] = {self.sequence.id:self.sequence.to_json()}
        datafiles = DataFile.objects.filter(entry=self.id)

        try:
            backup_root = config.backup_root['root']
        except:
            backup_root = '/None'
        
        js['datafiles'] = dict()
        system_names = set(d.system.name for d in datafiles)
        for name in system_names:
            js['datafiles'][name] = [d.get_path() + ' (backup available: %s)' % d.is_backed_up(backup_root) for d in datafiles if d.system.name == name]

        js['datafiles']['sequence'] = issubclass(Exp, experiment.Sequence) and len(self.sequence.sequence) > 0
        
        try:
            task = self.task.get(self.feats.all())
            report = json.loads(self.report)
            js['report'] = self.offline_report()
        except:
            import traceback
            traceback.print_exc()
            js['report'] = dict()

        # import config
        if config.recording_sys['make'] == 'plexon':
            try:
                from plexon import plexfile
                plexon = System.objects.get(name='plexon')
                df = DataFile.objects.get(entry=self.id, system=plexon)

                plx = plexfile.openFile(str(df.get_path()), load=False)
                path, name = os.path.split(df.get_path())
                name, ext = os.path.splitext(name)

                js['bmi'] = dict(_neuralinfo=dict(
                    length=plx.length, 
                    units=plx.units, 
                    name=name,
                    is_seed=int(Exp.is_bmi_seed),
                    ))
            except MemoryError:
                print "Memory error opening plexon file!"
                js['bmi'] = dict(_neuralinfo=None)
            except (ObjectDoesNotExist, AssertionError, IOError):
                print "No plexon file found"
                js['bmi'] = dict(_neuralinfo=None)
        
        elif config.recording_sys['make'] == 'blackrock':
            try:
                nev_fname = self.nev_file
                path, name = os.path.split(nev_fname)
                name, ext = os.path.splitext(name)

                #### start -- TODO: eventually put this code in helper functions somewhere else
                # convert .nev file to hdf file using Blackrock's n2h5 utility (if it doesn't exist already)
                # this code goes through the spike_set for each channel in order to:
                #  1) determine the last timestamp in the file
                #  2) create a list of units that had spikes in this file
                nev_hdf_fname = nev_fname + '.hdf'

                if not os.path.isfile(nev_hdf_fname):
                    import subprocess
                    subprocess.call(['n2h5', nev_fname, nev_hdf_fname])
                
                import h5py
                nev_hdf = h5py.File(nev_hdf_fname, 'r')

                last_ts = 0
                units = []

                for key in [key for key in nev_hdf.get('channel').keys() if 'channel' in key]:
                    if 'spike_set' in nev_hdf.get('channel/' + key).keys():
                        spike_set = nev_hdf.get('channel/' + key + '/spike_set')
                        if spike_set is not None:
                            tstamps = spike_set.value['TimeStamp']
                            if len(tstamps) > 0:
                                last_ts = max(last_ts, tstamps[-1])

                            channel = int(key[-5:])
                            for unit_num in np.sort(np.unique(spike_set.value['Unit'])):
                                units.append((channel, int(unit_num)))

                fs = 30000.
                nev_length = last_ts / fs

                nsx_fs = dict()
                nsx_fs['.ns1'] = 500
                nsx_fs['.ns2'] = 1000
                nsx_fs['.ns3'] = 2000
                nsx_fs['.ns4'] = 10000
                nsx_fs['.ns5'] = 30000
                nsx_fs['.ns6'] = 30000

                NSP_channels = np.arange(128) + 1

                nsx_lengths = []
                for nsx_fname in self.nsx_files:

                    nsx_hdf_fname = nsx_fname + '.hdf'
                    if not os.path.isfile(nsx_hdf_fname):
                        # convert .nsx file to hdf file using Blackrock's n2h5 utility
                        subprocess.call(['n2h5', nsx_fname, nsx_hdf_fname])

                    nsx_hdf = h5py.File(nsx_hdf_fname, 'r')

                    for chan in NSP_channels:
                        chan_str = str(chan).zfill(5)
                        path = 'channel/channel%s/continuous_set' % chan_str
                        if nsx_hdf.get(path) is not None:
                            last_ts = len(nsx_hdf.get(path).value)
                            fs = nsx_fs[nsx_fname[-4:]]
                            nsx_lengths.append(last_ts / fs)
                            
                            break

                length = max([nev_length] + nsx_lengths)
                #### end

                # Blackrock units start from 0 (unlike plexon), so add 1
                # for web interface purposes
                # i.e., unit 0 on channel 3 will be "3a" on web interface
                units = [(chan, unit+1) for chan, unit in units]

                js['bmi'] = dict(_neuralinfo=dict(
                    length=length, 
                    units=units, 
                    name=name,
                    is_seed=int(Exp.is_bmi_seed),
                    ))    
            except (ObjectDoesNotExist, AssertionError, IOError):
                print "No blackrock files found"
                js['bmi'] = dict(_neuralinfo=None)
        else:
            raise Exception('Unrecognized recording_system!')


        for dec in Decoder.objects.filter(entry=self.id):
            js['bmi'][dec.name] = dec.to_json()
        
        # include paths to any plots associated with this task entry, if offline
        files = os.popen('find /storage/plots/ -name %s*.png' % self.id)
        plot_files = dict()
        for f in files:
            fname = f.rstrip()
            keyname = os.path.basename(fname).rstrip('.png')[len(str(self.id)):]
            plot_files[keyname] = os.path.join('/static', fname)

        # if the juice log feature is checked, also include the snapshot of the juice if it exists
        try:
            juice_sys = System.objects.get(name='juice_log')
            df = DataFile.objects.get(system=juice_sys, entry=self.id)
            plot_files['juice'] = os.path.join(df.system.path, df.path)
        except DataFile.DoesNotExist:
            pass
        except:
            import traceback
            traceback.print_exc()

        js['plot_files'] = plot_files

        return js

    @property
    def plx_file(self):
        '''
        Returns the name of the plx file associated with the session.
        '''
        plexon = System.objects.get(name='plexon')
        try:
            df = DataFile.objects.get(system=plexon, entry=self.id)
            return os.path.join(df.system.path, df.path)
        except:
            import traceback
            traceback.print_exc()
            return 'noplxfile'


    @property
    def nev_file(self):
        '''
        Return the name of the nev file associated with the session.
        '''
        blackrock = System.objects.get(name='blackrock')
        q = DataFile.objects.filter(entry_id=self.id).filter(system_id=blackrock.id).filter(path__endswith='.nev')
        if len(q)==0:
            return 'nonevfile'
        else:
            try:
                import db.paths
                return os.path.join(db.paths.data_path, blackrock.name, q[0].path)
            except:
                return q[0].path

    @property
    def nsx_files(self):
        '''Return a list containing the names of the nsx files (there could be more
        than one) associated with the session.
        '''
        blackrock = System.objects.get(name='blackrock')
        q = DataFile.objects.filter(entry_id=self.id).filter(system_id=blackrock.id).exclude(path__endswith='.nev')
        if len(q)==0:
            return []
        else:
            try:
                import db.paths
                return [os.path.join(db.paths.data_path, blackrock.name, datafile.path) for datafile in q]
            except:
                return [datafile.path for datafile in q]


    @property
    def name(self):
        '''
        Return a string representing the 'name' of the block. Note that the block
        does not really have a unique name in the current implementation.
        Thus, the 'name' is a hack this needs to be hacked because the current way of determining a 
        a filename depends on the number of things in the database, i.e. if 
        after the fact a record is removed, the number might change. read from
        the file instead
        '''
        # import config
        if config.recording_sys['make'] == 'plexon':
            try:
                return str(os.path.basename(self.plx_file).rstrip('.plx'))
            except:
                return 'noname'
        elif config.recording_sys['make'] == 'blackrock':
            try:
                return str(os.path.basename(self.nev_file).rstrip('.nev'))
            except:
                return 'noname'
        else:
            raise Exception('Unrecognized recording_system!')

    @classmethod
    def from_json(cls, js):
        pass

    def get_decoder(self):
        """
        Get the Decoder instance associated with this task entry
        """
        params = eval(self.params)
        decoder_id = params['bmi']
        return Decoder.objects.get(id=decoder_id)

class Calibration(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    system = models.ForeignKey(System)

    params = models.TextField()

    def __unicode__(self):
        return "{date}:{system} calibration for {subj}".format(date=self.date, 
            subj=self.subject.name, system=self.system.name)
    
    def get(self):
        from json_param import Parameters
        return getattr(calibrations, self.name)(**Parameters(self.params).params)

class AutoAlignment(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.TextField()
    
    def __unicode__(self):
        return "{date}:{name}".format(date=self.date, name=self.name)
       
    def get(self):
        return calibrations.AutoAlign(self.name)

import importlib
def decoder_unpickler(mod_name, kls_name):
    if kls_name == 'StateSpaceFourLinkTentacle2D':
        kls_name = 'StateSpaceNLinkPlanarChain'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'StateSpaceEndptVel':
        kls_name = 'LinearVelocityStateSpace'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'State':
        mod_name = 'riglib.bmi.state_space_models'
    mod = importlib.import_module(mod_name)
    return getattr(mod, kls_name)


class Decoder(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    entry = models.ForeignKey(TaskEntry)
    path = models.TextField()
    
    def __unicode__(self):
        return "{date}:{name} trained from {entry}".format(date=self.date, name=self.name, entry=self.entry)
    
    @property 
    def filename(self):
        data_path = getattr(config, 'db_config_%s' % self._state.db)['data_path']
        return os.path.join(data_path, 'decoders', self.path)        

    def load(self,db_name=None):
        if db_name is not None:
            data_path = getattr(config, 'db_config_'+db_name)['data_path']
        else:
            data_path = getattr(config, 'db_config_%s' % self._state.db)['data_path']
        decoder_fname = os.path.join(data_path, 'decoders', self.path)

        # dec = pickle.load(open(decoder_fname))
        import cPickle
        fh = open(decoder_fname, 'r')
        unpickler = cPickle.Unpickler(fh)
        unpickler.find_global = decoder_unpickler
        dec = unpickler.load() # object will now contain the new class path reference
        fh.close()

        dec.name = self.name
        return dec        

    def get(self):
        return self.load()
        # sys = System.objects.get(name='bmi').path
        # return cPickle.load(open(os.path.join(sys, self.path)))

    def to_json(self):
        dec = self.get()
        return dict(
            name=self.name,
            cls=dec.__class__.__name__,
            path=self.path, 
            units=dec.units,
            binlen=dec.binlen,
            tslice=dec.tslice)

class DataFile(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    local = models.BooleanField(default=True)
    archived = models.BooleanField(default=False)
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System)
    entry = models.ForeignKey(TaskEntry)

    def __unicode__(self):
        return "{name} datafile for {entry}".format(name=self.system.name, entry=self.entry)

    def to_json(self):
        return dict(system=self.system.name, path=self.path)

    def get_path(self, check_archive=False):
        if not check_archive and not self.archived:
            return os.path.join(self.system.path, self.path)

        paths = self.system.archive.split()
        for path in paths:
            fname = os.path.join(path, self.path)
            if os.path.isfile(fname):
                return fname

        raise IOError('File has been lost! '+fname)

    def has_cache(self):
        if self.system.name != "plexon":
            return False

        path, fname = os.path.split(self.get_path())
        fname, ext = os.path.splitext(fname)
        cache = os.path.join(path, '.%s.cache'%fname)
        return os.path.exists(cache)

    def remove(self, **kwargs):
        try:
            os.unlink(self.get_path())
        except OSError:
            print "already deleted..."

    def delete(self, **kwargs):
        self.remove()
        super(DataFile, self).delete(**kwargs)

    def is_backed_up(self, backup_root):
        '''
        Return a boolean indicating whether a copy of the file is available on the backup
        '''
        fname = self.get_path()
        rel_datafile = os.path.relpath(fname, '/storage')
        backup_fname = os.path.join(backup_root, rel_datafile)
        return os.path.exists(backup_fname)
