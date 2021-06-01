# run with manage.py:
# > python manage.py shell
# > from bmi3d.db.migrate_db import *

from ..boot_django import boot_django
boot_django()

from .tracker import models
import os

def change_system_paths():
    '''
    Script to update system paths with a new ones. Applies to files in that system, too.
    '''
    optitrack_storage = "/media/Optitrack"
    ecube_storage = "/media/NeuroAcq"
    hdf_storage = "/home/pagaiisland/hdf"

    models.System.objects.filter(name='optitrack').update(path=optitrack_storage)
    models.System.objects.filter(name='ecube').update(path=ecube_storage)
    models.System.objects.filter(name='hdf').update(path=hdf_storage)

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

        if df.system.name == 'hdf':
            path = df.path
            (path, filename) = os.path.split(path)
            df.path = os.path.join(hdf_storage, filename)
            print(df.path)
            df.save()

def _make_seq_name(gen, params):
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

def fix_seq_names():
    ''' Utility to re-do all the sequence names '''
    for seq in models.Sequence.objects.all():
        json = seq.to_json()
        params = json['params']
        print(seq.name)
        seq.name = _make_seq_name(seq.generator.name, params)
        print(seq.name)
        print('-')
        seq.save()
    

import importlib

def _fix_import_path(import_path):
    try:
        importlib.import_module(import_path)
        return import_path
    except:
        import_path = "bmi3d." + import_path
        return import_path

def update_model_import_paths():
    ''' Script to re-do all the model import paths '''
    models_with_paths = list(models.Task.objects.all()) + list(models.Feature.objects.all())

    for model in models_with_paths:
        print(model.import_path)
        model.import_path = _fix_import_path(model.import_path)
        print(model.import_path)
        print('-')
        model.save()