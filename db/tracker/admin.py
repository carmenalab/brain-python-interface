'''
Declare which database tables are visible from Django's default admin interface. 

This file was initially created by Django
'''
from .models import Task, Feature, System, TaskEntry, Calibration, DataFile, Subject, Sequence, Generator, AutoAlignment, Decoder
from django.contrib import admin
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver

@receiver(post_delete, sender=DataFile)
def _mymodel_delete(sender, instance, **kwargs):
    instance.remove()
    
admin.site.register(Task)
admin.site.register(Feature)
admin.site.register(System)
admin.site.register(TaskEntry)
admin.site.register(Calibration)
admin.site.register(DataFile)
admin.site.register(Subject)
admin.site.register(Sequence)
admin.site.register(Generator)
admin.site.register(AutoAlignment)
admin.site.register(Decoder)
