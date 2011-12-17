from tracker.models import Task, Feature, System, TaskEntry, Calibration, DataFile, Subject
from django.contrib import admin

admin.site.register(Task)
admin.site.register(Feature)
admin.site.register(System)
admin.site.register(TaskEntry)
admin.site.register(Calibration)
admin.site.register(DataFile)
admin.site.register(Subject)