import json
import cPickle
import inspect
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

from riglib import calibrations
from riglib.experiment import featlist, genlist
from tasks import tasklist

class Task(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    def get(self):
        return tasklist[self.name]

    @staticmethod
    def populate():
        real = set(tasklist.keys())
        db = set(task.name for task in Task.objects.all())
        for name in real - db:
            Task(name=name).save()

        for name in db - real:
            Task.objects.get(name=name).delete()

class Feature(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    def get(self):
        return featlist[self.name]

    @staticmethod
    def populate():
        real = set(featlist.keys())
        db = set(feat.name for feat in Feature.objects.all())
        for name in real - db:
            Feature(name=name).save()

        for name in db - real:
            Feature.objects.get(name=name).delete()

class System(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    @staticmethod
    def populate():
        System(name="eyetracker").save()
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
        return genlist[self.name]

    @staticmethod
    def populate():
        real = set(genlist.keys())
        db = set(gen.name for gen in Generator.objects.all())
        for name in real - db:
            args = inspect.getargspec(genlist[name]).args
            static = "length" in args
            if "exp" in args:
                args.remove("exp")
            if "length" in args:
                args.remove("length")
            Generator(name=name, params=",".join(args), static=static).save()

        for name in db - real:
            Generator.objects.get(name=name).delete()

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
        from json_param import Parameters
        if len(self.sequence) > 0:
            return experiment.generate.runseq, cPickle.loads(self.sequence)
        return self.generator.get(), Parameters(self.params)


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
        Exp = experiment.make(self.task.get(), tuple(f.get() for f in self.feats.all())+feats)
        gen, gp = self.sequence.get()
        seq = gen(Exp, **gp)
        return Exp(seq, **Parameters(self.params).params)

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