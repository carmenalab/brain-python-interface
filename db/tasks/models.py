from django.db import models

class Tasks(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Features(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Varieties(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class TaskEntry(models.Model):
    task = models.ForeignKey(Tasks)
    feats = models.ManyToManyField(Features)
    params = models.TextField()
    notes = models.TextField()
    report = models.TextField()
    date = models.DateTimeField(auto_now_add=True)

class Calibration(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    notes = models.TextField()
    variety = models.ForeignKey(Varieties)
    data = models.TextField()