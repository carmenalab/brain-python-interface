## #!/usr/bin/python
## import os
## import sys
## 
## import django
## from distutils.version import LooseVersion, StrictVersion
## 
## if StrictVersion('%d.%d.%d' % django.VERSION[0:3]) < StrictVersion('1.6.0'):
##     from django.core.management import execute_manager
##     try:
##         from . import settings # Assumed to be in the same directory.
##     except ImportError:
##         import sys
##         sys.stderr.write("Error: Can't find the file 'settings.py' in the directory containing %r. It appears you've customized things.\nYou'll have to run django-admin.py, passing it your settings module.\n(If the file settings.py does indeed exist, it's causing an ImportError somehow.)\n" % __file__)
##         sys.exit(1)
##     
##     execute_manager(settings)
## else:
##     # configure celery
##     import celery
##     c = celery.Celery()
##     c.conf.update(CELERY_ACCEPT_CONTENT = ['json', 'pickle'])
## 
##     os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")
## 
##     from django.core.management import execute_from_command_line
## 
##     execute_from_command_line(sys.argv)

#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'db.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

