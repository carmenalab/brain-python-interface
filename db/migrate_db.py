# run with manage.py:
# > python manage.py shell
# > from db.migrate_db import *

from .boot_django import boot_django
boot_django()

from .tracker import models
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import django
django.setup()

from db.tracker import models

'''
Update system paths
'''
#models.System.objects.filter(name='optitrack').update(path='/media/Optitrack')
#models.System.objects.filter(name='ecube').update(path='/media/NeuroAcq')

# optitrack_storage = "/media/Optitrack/"
# ecube_storage = "/media/NeuroAcq"

# for df in models.DataFile.objects.all():
#     if df.system.name == 'optitrack':
#         path = df.path
#         (path, filename) = os.path.split(path)
#         (path, session) = os.path.split(path)
#         (path, base_dir) = os.path.split(path)
#         df.path = os.path.join(optitrack_storage, base_dir, session, filename + '.tak')
#         print(df.path)
#         df.save()

#     if df.system.name == 'ecube':
#         path = df.path
#         (path, filename) = os.path.split(path)
#         df.path = os.path.join(ecube_storage, filename)
#         print(df.path)
#         df.save()

'''
Fix the sequence names to be reflections of their parameters
'''
# def make_seq_name(gen, params):
#     param_str = []
#     for (k, v) in params.items():
#         value_str = []
#         try:
#             for x in v['value']:
#                 value_str.append(str(x))
#         except:
#             value_str.append(str(v['value']))
#         param_str.append(f"{k}={','.join(value_str)}")
#     return f"{gen}:[{', '.join(param_str)}]"

# for seq in models.Sequence.objects.all():
#     json = seq.to_json()
#     params = json['params']
#     print(seq.name)
#     seq.name = make_seq_name(seq.generator.name, params)
#     print(seq.name)
#     print('-')
#     seq.save()

'''
Add metadata from xls files
'''
import pandas as pd
from datetime import datetime
df = pd.read_csv("/home/pagaiisland/Downloads/Beignet Records - Training 2021.csv", header=1)
handler_dict = {}
for index, row in df.iterrows():
    date = row["Date"]
    fdate = None
    if type(date) is str and date != '#REF!':
        try:
            fdate = datetime.strptime(date, "%m/%d/%Y")
        except:
            fdate = datetime.strptime(date+"/2021", "%m/%d/%Y")
    
    if fdate and not pd.isna(row["Handler"]):
        handler = str(row["Handler"]).lower()
        if "leo" in handler:
            handler = "Leo"
        elif "pavi" in handler:
            handler = "Pavi"
        elif "lydia" in handler:
            handler = "Lydia"
        elif "ryan" in handler:
            handler = "Ryan"
        else:
            print(handler)
            print(fdate)
            handler = "Leo"

        if fdate.date() not in handler_dict:
            handler_dict[fdate.date()] = dict.fromkeys([handler])
        else:
            handler_dict[fdate.date()].update(dict.fromkeys([handler]))

#print(handler_dict)

for te in models.TaskEntry.objects.all():
    date = te.date.date()
    handler = None
    project = "testing"
    session = ""
    if date in handler_dict and te.subject.name == "beignet":
        handlers = list(handler_dict[date])
        if len(handlers) == 1:
            handler = handlers[0]
        elif te.date.time() < datetime(year=2021,month=1,day=1,hour=13).time():
            handler = handlers[0]
            session = "morning"
        else:
            handler = handlers[1]
            session = "afternoon"

        if te.date.date() < datetime(year=2021, month=3, day=29).date():
            project = "open loop training"
        else:
            project = "manual control training"
    elif te.subject.name == "affi":
        handler = "Lydia"
        project = "open loop training"
    elif te.subject.name == "test":
        handler = "Leo"
    else:
        handler = "Lydia"
        
    experimenter = models.Experimenter.objects.get(name=handler)
    
    te.experimenter = experimenter
    te.project = project
    te.session = session
    te.save()
