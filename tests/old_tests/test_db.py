import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'db.settings')
django.setup()

from db.tracker import models

fe = models.Feature.objects.all()
print(fe)

ID_NUMBER = 77
te = models.TaskEntry.objects.get(id=ID_NUMBER)
print(te.subject)


# try a different method
# need to take out the dot in the .tracker import
# this method gets depreciated 
import db.dbfunctions as dbfn
te_1 = dbfn.TaskEntry(ID_NUMBER)
print(te_1.date)

'''

'''