import json
import cPickle
from django.db import models

from riglib.calibrations import ThinPlate
from riglib.experiment import featlist, genlist
from tasks import tasklist

class Task(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    def get(self):
        return tasklist[self.name]

class Feature(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name
    
    def get(self):
        return featlist[self.name]

class System(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

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
        if len(self.sequence) > 0:
            return experiment.generate.runseq(cPickle.loads(self.sequence))
        return self.generator.get(), json.loads(self.params)


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

class Calibration(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    system = models.ForeignKey(System)

    params = models.TextField()

    def __unicode__(self):
        return self.name
    
    def get(self):
        return ThinPlate(**json.loads(self.params))


class DataFile(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    local = models.BooleanField()
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System)
    entruy = models.ForeignKey(TaskEntry)