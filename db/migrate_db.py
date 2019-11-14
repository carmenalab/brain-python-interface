# run with manage.py shell

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import django
django.setup()

from tracker import models

for te in models.TaskEntry.objects.all():
    if te.visible is None:
        te.visible = False
        te.save()

    if te.backup is None:
        te.backup = False
        te.save()

    if te.sequence_id == -1:
        te.sequence_id = 1
        print("TaskEntry sequence ID was invalid: ", te.id)
        te.save()

for df in models.DataFile.objects.all():
    if df.entry_id == -1:
        df.entry_id = 1
        print("DataFile task entry is invalid: ", df.id)
        df.save()
