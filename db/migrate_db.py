# run with manage.py:
# > python manage.py shell
# > import update_db_paths

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import django
django.setup()

from db.tracker import models

#models.System.objects.filter(name='optitrack').update(path='/media/Optitrack')
#models.System.objects.filter(name='ecube').update(path='/media/NeuroAcq')

optitrack_storage = "/media/Optitrack/"
ecube_storage = "/media/NeuroAcq"

for df in models.DataFile.objects.all():
    if df.system.name == 'optitrack':
        path = df.path
        (path, filename) = os.path.split(path)
        (path, session) = os.path.split(path)
        (path, base_dir) = os.path.split(path)
        df.path = os.path.join(optitrack_storage, base_dir, session, filename + '.tak')
        print(df.path)
        df.save()

    if df.system.name == 'ecube':
        path = df.path
        (path, filename) = os.path.split(path)
        df.path = os.path.join(ecube_storage, filename)
        print(df.path)
        df.save()
