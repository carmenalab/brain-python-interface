# run with manage.py:
# > python manage.py shell
# > import update_db_paths

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'bmi3d.db.settings'
import django
django.setup()

from .tracker import models

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

def make_seq_name(gen, params):
    param_str = []
    for (k, v) in params.items():
        value_str = []
        try:
            for x in v['value']:
                value_str.append(str(x))
        except:
            value_str.append(str(v['value']))
        param_str.append(f"{k}={','.join(value_str)}")
    return f"{gen}:[{', '.join(param_str)}]"

for seq in models.Sequence.objects.all():
    json = seq.to_json()
    params = json['params']
    print(seq.name)
    seq.name = make_seq_name(seq.generator.name, params)
    print(seq.name)
    print('-')
    seq.save()