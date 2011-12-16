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

class TaskEntry(models.Model):
    task = models.ForeignKey(Task)
    feats = models.ManyToManyField(Feature)
    params = models.TextField()
    notes = models.TextField()
    report = models.TextField()
    date = models.DateTimeField(auto_now_add=True)

class Calibration(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    notes = models.TextField()
    system = models.ForeignKey(System)
    data = models.TextField()