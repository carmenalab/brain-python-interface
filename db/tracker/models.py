'''
Django database modules. See https://docs.djangoproject.com/en/dev/intro/tutorial01/
for a basic introduction
'''

import os
import json
import cPickle
import inspect
from collections import OrderedDict
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

import numpy as np

from riglib import calibrations, experiment

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
    visible = models.BooleanField(default=True)
    def __unicode__(self):
        return self.name
    
    def get(self, feats=()):
        from namelist import tasks
        from riglib import experiment
        return experiment.make(tasks[self.name], Feature.getall(feats))

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
            if trait in values:
                varname['value'] = values[trait]
            if varname['type'] == "Instance":
                Model = instance_to_model[ctraits[trait].trait_type.klass]
                insts = Model.objects.order_by("-date")[:50]
                varname['options'] = [(i.pk, i.name) for i in insts]
            params[trait] = varname

        ordered_traits = Exp.ordered_traits
        for trait in ordered_traits:
            if trait in Exp.class_editable_traits():
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
    visible = models.BooleanField()
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
        for name in ["eyetracker", "hdf", "plexon", "bmi", "bmi_params"]:
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
    visible = models.BooleanField()

    def __unicode__(self):
        return self.name
    
    def get(self):
        from namelist import generators
        return generators[self.name]

    @staticmethod
    def populate():
        from namelist import generators
        real = set(generators.keys())
        db = set(gen.name for gen in Generator.objects.all())
        for name in real - db:
            try:
                args = inspect.getargspec(generators[name]).args
            except TypeError:
                args = inspect.getargspec(generators[name].__init__).args
                args.remove("self")
            
            static = "length" in args
            if "exp" in args:
                args.remove("exp")
            if "length" in args:
                args.remove("length")
            Generator(name=name, params=",".join(args), static=static).save()

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

        if self.static:
            defaults = (None,)+defaults
        else:
            #first argument is the experiment
            names.remove("exp")
        arginfo = zip(names, defaults)

        params = dict()
        for name, default in arginfo:
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
            if len(self.sequence) > 0:
                return generate.runseq, dict(seq=cPickle.loads(str(self.sequence)))
            return generate.runseq, dict(seq=self.generator.get()(**Parameters(self.params).params))

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
        from json_param import Parameters
        try:
            return Sequence.objects.get(pk=int(js))
        except:
            pass
        
        if not isinstance(js, dict):
            js = json.loads(js)
        genid = js['generator']
        if isinstance(genid, (tuple, list)):
            genid = genid[0]
        
        seq = cls(generator_id=int(genid), name=js['name'])
        seq.params = Parameters.from_html(js['params']).to_json()
        if js['static']:
            seq.sequence = cPickle.dumps(seq.generator.get()(**Parameters(seq.params).params))
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

    def to_json(self):
        from json_param import Parameters
        Exp = self.task.get(self.feats.all())

        state = 'completed' if self.pk is not None else "new"
        js = dict(task=self.task.id, state=state, subject=self.subject.id, notes=self.notes)
        js['feats'] = dict([(f.id, f.name) for f in self.feats.all()])
        js['params'] = self.task.params(self.feats.all(), values=Parameters(self.params).params)
        if issubclass(self.task.get(), experiment.Sequence):
            js['sequence'] = {self.sequence.id:self.sequence.to_json()}
        datafiles = DataFile.objects.filter(entry=self.id)
        js['datafiles'] = dict([(d.system.name, os.path.join(d.system.path,d.path)) for d in datafiles])
        js['datafiles']['sequence'] = issubclass(Exp, experiment.Sequence) and len(self.sequence.sequence) > 0
        try:
            task = self.task.get(self.feats.all())
            report = json.loads(self.report)
            js['report'] = Exp.offline_report(report)
        except:
            import traceback
            traceback.print_exc()
            js['report'] = dict()

        try:
            from plexon import plexfile
            plexon = System.objects.get(name='plexon')
            df = DataFile.objects.get(entry=self.id, system=plexon)

            plx = plexfile.openFile(str(df.get_path()), load=False)
            path, name = os.path.split(df.get_path())
            name, ext = os.path.splitext(name)

            from namelist import bmi_seed_tasks
            js['bmi'] = dict(_plxinfo=dict(
                length=plx.length, 
                units=plx.units, 
                name=name,
                is_seed=int(self.task.name in bmi_seed_tasks),
                ))
        except (ObjectDoesNotExist, AssertionError, IOError):
            print "No plexon file found"
            js['bmi'] = dict(_plxinfo=None)

        for dec in Decoder.objects.filter(entry=self.id):
            js['bmi'][dec.name] = dec.to_json()
        
        # TODO include paths to any plots associated with this task entry, if offline
        files = os.popen('find /storage/plots/ -name %s*.png' % self.id)
        plot_files = dict()
        for f in files:
            fname = f.rstrip()
            keyname = os.path.basename(fname).rstrip('.png')[len(str(self.id)):]
            plot_files[keyname] = os.path.join('/static', fname)
        js['plot_files'] = plot_files

        print js['report'].keys()
        return js

    @property
    def plx_file(self):
        '''
        Returns the name of the plx file associated with the session.
        '''
        plexon = System.objects.get(name='plexon')
        q = DataFile.objects.filter(entry_id=self.id).filter(system_id=plexon.id)
        if len(q)==0:
            return 'noplxfile'
        else:
            try:
                import db.paths
                return os.path.join(db.paths.data_path, plexon.name, q[0].path)
            except:
                return q[0].path

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
        try:
            return str(os.path.basename(self.plx_file).rstrip('.plx'))
        except:
            return 'noname'



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

class Decoder(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    entry = models.ForeignKey(TaskEntry)
    path = models.TextField()
    
    def __unicode__(self):
        return "{date}:{name} trained from {entry}".format(date=self.date, name=self.name, entry=self.entry)
    
    def get(self):
        sys = System.objects.get(name='bmi').path
        return cPickle.load(open(os.path.join(sys, self.path)))

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
    local = models.BooleanField()
    archived = models.BooleanField()
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

        raise IOError('File has been lost!')

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
