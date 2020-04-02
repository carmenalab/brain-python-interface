# run with manage.py shell

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import django
django.setup()

from tracker import models

task_entry_with_invalid_sequence_id = []

for te in models.TaskEntry.objects.all():
    if te.visible is None:
        te.visible = False
        te.save()

    if te.backup is None:
        te.backup = False
        te.save()

    if te.sequence_id == -1:
        te.sequence_id = 1
        te.save()
        task_entry_with_invalid_sequence_id.append(te.id)

print("TaskEntry sequence ID was invalid")
print(task_entry_with_invalid_sequence_id)

df_with_invalid_task_entry_id = []
for df in models.DataFile.objects.all():
    if df.entry_id == -1:
        df.entry_id = 1
        df.save()
        df_with_invalid_task_entry_id.append(df.id)

print("DataFile task entry is invalid: ", df.id)
print(df_with_invalid_task_entry_id)
