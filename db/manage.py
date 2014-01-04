#!/usr/bin/python
import os
import sys

import django
from distutils.version import LooseVersion, StrictVersion

if StrictVersion('%d.%d.%d' % django.VERSION[0:3]) < StrictVersion('1.6.0'):
    from django.core.management import execute_manager
    try:
        import settings # Assumed to be in the same directory.
    except ImportError:
        import sys
        sys.stderr.write("Error: Can't find the file 'settings.py' in the directory containing %r. It appears you've customized things.\nYou'll have to run django-admin.py, passing it your settings module.\n(If the file settings.py does indeed exist, it's causing an ImportError somehow.)\n" % __file__)
        sys.exit(1)
    
    execute_manager(settings)
else:
    # configure celery
    import celery
    c = celery.Celery()
    c.conf.update(CELERY_ACCEPT_CONTENT = ['json'])

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
