# boot_django.py
#
# This file sets up and configures Django. It's used by scripts that need to
# execute as if running in a Django server.
import django
from django.conf import settings
from db import db_settings

def get_module_settings(module_name):
    module = globals().get(module_name, None)
    settings = {}
    if module:
        settings = {key: value for key, value in module.__dict__.items() \
            if not (key.startswith('__') or key.startswith('_')) and key.isupper()}
    return settings

def boot_django():
    if not settings.configured:
        overrides = get_module_settings('db_settings')
        settings.configure(**overrides)
        django.setup()