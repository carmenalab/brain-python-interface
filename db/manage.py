#!/usr/bin/python
import os
import sys
if __name__ == "__main__":
    # configure celery
    import celery
    c = celery.Celery()
    c.conf.update(CELERY_ACCEPT_CONTENT = ['json'])

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "db.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
