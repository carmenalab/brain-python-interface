.. _analysis:

Analyzing Data
==============

If you have run all your experiments through the browser, then all your experimental data is stored in the database. This means all you need to keep track of is the id number of the database records you actually want to analyze. No need to remember/copy-paste all 800 filenames associated with any one experiment, you can just look them up in the database! The modules described in this document describe tools available for making data and metadata retrieval more pleasant. This document is NOT meant to be a comprehensive descriptor of all the conceivable analyses you could do with your data. Because that's your job. 


Retrieving the metadata
-----------------------
Suppose you have a task ``Task1`` with runtime-configurable parameters ``paramA`` and ``paramB``. The most barebones way to get back the specific values of ``paramA`` and ``paramB`` used during the task is the following::

    import os
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from db.tracker import models
    te = models.TaskEntry.objects.get(id=ID_NUMBER)
    te.params['paramA']

Most tasks will typically have at least one file linked as well to store all the valuable data generated during the actual experiment. Most commonly an HDF file will be linked to the task entry as well. To get the name of the HDF file, you could do::

    sys_ = models.System.objects.get(name='SaveHDF')
    df = models.DataFile.objects.get(id=te.id, system=sys_)
    hdf_filename = os.path.join(sys_.path, df.path)
    import tables
    hdf = tables.open_file(hdf_filename)

This is a perfectly valid way to get the data, but will get old in a hurry. The purpose of the ``db.dbfunctions`` module is to reduce the number of steps required to accomplish simple database tasks::

    from db import dbfunctions as dbfn
    te = dbfn.TaskEntry(ID_NUMBER)
    paramA = te.paramA
    hdf = te.hdf

In essence, the purpose of the class ``db.dbfunctions.TaskEntry`` is to shield the experimenter from low-level database access commands. Django provides one layer of abstraction so that you do not have to write sqlite3 commands. ``db.dbfunctions`` provides another layer so that common associations between different database tables and external files are handled implicitly. It's not perfect, of course.


Using multiple databases
------------------------
.. note :: Multiple databases should be used for analysis machines only. Although it should work okay on rig machines, it may also confuse which database it is writing to. You have been warned!

When analyzing data from multiple rigs, you often want to use multiple databases if subjects were run in different systems, etc. Django also makes it possible to use multiple databases: https://docs.djangoproject.com/en/dev/topics/db/multi-db/. To be able to seamlessly read from multiple databases, the file ``$BMI3D/db/settings.py`` must be told about the other database files. For example,::

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
            'NAME': os.path.join(cwd, "db.sql"),                      # Or path to database file if using sqlite3.
            'USER': '',                      # Not used with sqlite3.
            'PASSWORD': '',                  # Not used with sqlite3.
            'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
            'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
        },
        'bmi3d': {
            'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
            'NAME': os.path.join(cwd, "db_bmi3d.sql"),                      # Or path to database file if using sqlite3.
            'USER': '',                      # Not used with sqlite3.
            'PASSWORD': '',                  # Not used with sqlite3.
            'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
            'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
        },
        'exorig': {
            'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
            'NAME': os.path.join(cwd, "db_exorig.sql"),                      # Or path to database file if using sqlite3.
            'USER': '',                      # Not used with sqlite3.
            'PASSWORD': '',                  # Not used with sqlite3.
            'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
            'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
        },
        'simulation': {
            'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
            'NAME': os.path.join(cwd, "db_simulation.sql"),                      # Or path to database file if using sqlite3.
            'USER': '',                      # Not used with sqlite3.
            'PASSWORD': '',                  # Not used with sqlite3.
            'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
            'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
        }
    }

specifies three different databases which can be used. Different databases of course
must have unique names. Each name specifies a path to a .sql file (note that in each sub-dictionary, a different sqlite3 database is specified). 

The file ``settings.py`` is version-controlled, so that each new clone of the repository has the standard configuration. However, local machine-specific changes should not be pushed upstream, so that everyone can happily maintain their own settings.py file without worrying about their configurations getting overwritten by other people's changes. So, after you've finished modifying settings.py, execute the shell commands::

    cd $BMI3D
    git update-index --skip-worktree db/settings.py

The ``skip-worktree`` directive indicates to git to ignore local changes to the file. You can find more information at http://stackoverflow.com/questions/13630849/git-difference-between-assume-unchanged-and-skip-worktree

The package will now need to be "reconfigured" to know about the new databases. Reconfiguration can be performed by running the script ``$BMI3D/make_config.py``, which will (re)generate ``$BMI3D/config``. 

To continue using the ``dbfn.TaskEntry`` setup::

    te = dbfn.TaskEntry(ID_NUMBER, dbname=DATABASE_NAME)

where ``DATABASE_NAME`` is of one of the possible databases you listed in settings.py. If this keyword argument is not specified, ``dbfn.TaskEntry`` will use the 'default' database.



Actually analyzing the data
---------------------------
[[This section is EXTREMELY INCOMPLETE]]

The high-level flow of any analysis is 

    1. collect the ID numbers of blocks you want to process, either manually or using database filters
    2. group the blocks by some criterion, e.g., date, type of BMI decoder used, task parameters.
    3. analyze the data


Grouping the data blocks
""""""""""""""""""""""""
Often different data blocks will not be independent and should be treated as a single block for "analysis" purposes. For example, if you want to calculate the performance of a particular set of decoder parameters, and you have two BMI blocks from the same day that used the same parameters, then for the purposes of analysis you may want to treat the data as having come from one big block. Once you have identified which blocks of data you want to analyze, this function can help group them:

.. automodule :: db.dbfunctions
    :members: group_ids

By default, the grouping is by date, as this is the most common use case that we have encountered.

Processing data blocks
""""""""""""""""""""""


.. autoclass:: db.dbfunctions.TaskEntry
    :members: proc

Note that dbfunctions.TaskEntry does not inherit from ``db.tracker.models.TaskEntry`` because inhertinging from Django models is a little more involved than inheriting from a normal python class. Instead, dbfunctions.TaskEntry is a wrapper for db.tracker.models.TaskEntry. 

Instantiating a dbfunctions.TaskEntry object with a database ID number creates an object with basic attributes, e.g., names of linked HDF/plexon files. However, we will often want to perform analyses that are only meaningful for the particular task of a block. For this, we create task-specific child classes, one for each task. These are declared in the semi-misnamed analysis.performance module. 

To get one of these task-specific TaskEntry classes, you can do

.. code-block:: python

    from bmi3d import analysis
    te = analysis.get(id)
