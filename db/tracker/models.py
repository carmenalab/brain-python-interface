from django.db import models

class Task(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Feature(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class System(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Subject(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class TaskEntry(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    task = models.ForeignKey(Task)
    feats = models.ManyToManyField(Feature)
    params = models.TextField()
    report = models.TextField()
    notes = models.TextField()

class Calibration(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    notes = models.TextField()
    system = models.ForeignKey(System)
    path = models.CharField(max_length=256)

class DataFile(models.Model):
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System)
    taskentry = models.ForeignKey(TaskEntry)