import json
import cPickle
import inspect
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

from riglib import calibrations

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
    def __unicode__(self):
        return self.name
    
    def get(self):
        from namelist import tasks
        return tasks[self.name]

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

        if len(feats) > 0 and isinstance(feats[0], (int, float, str)):
            feats = tuple(Feature.objects.get(pk=int(f)) for f in feats)
        
        params = dict()
        Exp = experiment.make(self.get(), tuple(f.get() for f in feats))
        ctraits = Exp.class_traits()
        for trait in Exp.class_editable_traits():
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

        return params

    def sequences(self):
        from json_param import Parameters
        seqs = dict()
        for s in Sequence.objects.filter(task=self.id):
            seqs[s.id] = s.to_json()
        
        return seqs

    @classmethod
    def from_json(cls, js):
        pass

class Feature(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
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

class System(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    @staticmethod
    def populate():
        try:
            System.objects.get(name="eyetracker")
        except:
            System(name="eyetracker").save()
        try:
            System.objects.get(name="motiontracker")
        except:
            System(name="motiontracker").save()

class Subject(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Generator(models.Model):
    name = models.CharField(max_length=128)
    params = models.TextField()
    static = models.BooleanField()

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
        js['params'] = Parameters(self.params).params
        js['generator'] = self.generator.id, self.generator.name
        return js

    @classmethod
    def from_json(cls, js):
        if not isinstance(js, dict):
            js = json.loads(js)
        assert js['state'] == "new"
        genid = js['gen']
        if isinstance(genid, (tuple, list)):
            genid = genid[0]
        
        seq = cls(generator_id=genid, name=js['name'], params=json.dumps(js['params']))
        if js['static']:
            seq.sequence = cPickle.dumps(seq.generator.get()(**Parameters(self.params).params))
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
        return "{date}: {subj} on {task} task".format(
            date=self.date.strftime("%h. %e, %Y, %l:%M %p"),
            subj=self.subject.name,
            task=self.task.name)
    
    def get(self, feats=()):
        from json_param import Parameters
        from riglib import experiment
        Exp = experiment.make(self.task.get(), tuple(f.get() for f in self.feats.all())+feats)
        gen, gp = self.sequence.get()
        seq = gen(Exp, **gp.params)
        exp = Exp(seq, **Parameters(self.params).params)
        exp.event_log = json.loads(self.report)
        return exp

    def to_json(self):
        from json_param import Parameters
        state = 'completed' if self.pk is not None else "new"
        js = dict(task=self.task.name, state=state, params=dict())
        js['feats'] = dict([(f.id, f.name) for f in self.feats.all()])
        js['params'] = self.task.params(self.feats.all(), values=Parameters(self.params).params)
        if self.sequence_id > 0:
            js['seq'] = {self.sequence.id:self.sequence.to_json()}
            
        return js

    @classmethod
    def from_json(cls, js):
        pass

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


class DataFile(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    local = models.BooleanField()
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System)
    entry = models.ForeignKey(TaskEntry)
