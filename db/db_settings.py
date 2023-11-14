'''
Django config file mostly auto-generated when a django project is created.
See https://docs.djangoproject.com/en/dev/intro/tutorial01/ for an introduction
on how to customize this file test
'''
import os, glob, re
import glob
import re
import socket
import json
from django.core.exceptions import ImproperlyConfigured

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(os.path.expanduser("~"), '.config', 'bmi3d')
HOSTNAME = socket.gethostname()

def get_sqlite3_databases():
    dbs = dict()
    db_files = glob.glob(os.path.join(BASE_DIR, '*.sql'))
    for db in db_files:
        db_name_re = re.match('db(.*?).sql', os.path.basename(db))
        db_name = db_name_re.group(1)
        print(db_name)
        if db_name.startswith('_'):
            db_name = db_name[1:]
        elif db_name == "":
            db_name = "default"
        else:
            # unrecognized db name pattern
            print("Unrecognized database file: ", db)
            continue

        dbs[db_name] = {
            'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or'oracle'.
            'NAME': db,                      # Or path to database file if using sqlite3.
            'USER': '',                      # Not used with sqlite3.
            'PASSWORD': '',                  # Not used with sqlite3.
            'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
            'PORT': '',                      # Set to empty string for default. Not used with sqlite3.        
        }

    if len(dbs.keys()) == 1 and 'test_aopy' in dbs.keys():
        dbs = {'default': dbs['test_aopy']}

    return dbs


# DATABASES = { 'default': { 'ENGINE': 'django.db.backends.sqlite3', 'NAME': 'db.sql', } }
DATABASES = {}

def get_secret(setting):
    """Get secret setting or fail with ImproperlyConfigured"""
    try:
        with open(os.path.join(CONFIG_DIR, 'secrets.json')) as secrets_file:
            secrets = json.load(secrets_file)
        return secrets[setting]
    except KeyError:
        raise ImproperlyConfigured("Set the {} setting".format(setting))

def get_mysql_database(dbname):
    return {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': dbname,
        'HOST': get_secret("DB_HOST"),
        'PORT': get_secret("DB_PORT"),
        'USER': get_secret("DB_USER"),
        'PASSWORD': get_secret("DB_PASSWORD")
    }

if HOSTNAME in ['pagaiisland2']:
    DATABASES['default'] = get_mysql_database('rig1')
    CELERY_BROKER_URL = f'amqp://{get_secret("AMQP_USER")}:{get_secret("AMQP_PASSWORD")}@{get_secret("AMQP_HOST")}:{get_secret("AMQP_PORT")}//'
elif HOSTNAME in ['siberut-bmi']:
    DATABASES['default'] = get_mysql_database('rig2')
    CELERY_BROKER_URL = f'amqp://{get_secret("AMQP_USER")}:{get_secret("AMQP_PASSWORD")}@{get_secret("AMQP_HOST")}:{get_secret("AMQP_PORT")}//'
elif HOSTNAME in ['booted-server']:
    DATABASES['default'] = get_mysql_database('tablet')
elif HOSTNAME in ['moor', 'crab-eating', 'ecube']:
    DATABASES['rig1'] = get_mysql_database('rig1')
    DATABASES['rig2'] = get_mysql_database('rig2')
    DATABASES['tablet'] = get_mysql_database('tablet')
    DATABASES['default'] = DATABASES['rig1']
else:
    DATABASES = get_sqlite3_databases()

# Django settings for db project.
DEBUG = True
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@domain.com'),
)

MANAGERS = ADMINS

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'America/Los_Angeles'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale
USE_L10N = True

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = ''

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = 'i$k&ifgz&_=m+!(1dm*9g3a1v6ue6#10_r!j!y6g^oj=1ha_z!'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
)

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


ROOT_URLCONF = 'db.urls'

TEMPLATE_DIRS = (
    os.path.join(BASE_DIR, "html", "templates"),
)

STATIC_URL = "/static/"

STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "html", "static"),
    "/usr/share/pyshared/django/contrib/"
)
ADMIN_MEDIA_PREFIX = '/static/admin/'
INSTALLED_APPS = (
    'db.tracker',
    'django.contrib.auth',
    'django.contrib.admin',    
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
)

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.jinja2.Jinja2',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'environment': 'db.tracker.jinja2.environment'
        },
    },
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

LOGGING_CONFIG = None
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
}

APPEND_SLASH=False
ALLOWED_HOSTS = ['*'] #['127.0.0.1', 'localhost', "testserver"]


