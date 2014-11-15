.. _analysis:

Data analysis code
------------------

The high-level flow of any analysis is 

	1. collect the ID numbers of blocks you want to process, either manually or using database filters
	2. group the blocks by some criterion, e.g., date, type of BMI decoder used, task parameters.
	3. analyze the data

Collecting the data blocks
==========================

Grouping the data blocks
========================
Often different data blocks will not be independent and should be treated as a single block for "analysis" purposes. For example, if you want to calculate the performance of a particular set of decoder parameters, and you have two BMI blocks from the same day that used the same parameters, then for the purposes of analysis you may want to treat the data as having come from one big block. Once you have identified which blocks of data you want to analyze, this function can help group them:

.. automodule :: db.dbfunctions
	:members: group_ids

By default, the grouping is by date, as this is the most common use case that we have encountered.

Processing data blocks
======================
Experiments run with the BMI3D experimental software are automatically logged into a database using the python package django. Central to the analysis of experimental data is the TaskEntry
model in the Django database (db.tracker.TaskEntry). However, it's not desireable to be 
manipulating the database code frequently since it's critical to data recording, so we mostly put our analysis code in a different place:

.. autoclass:: db.dbfunctions.TaskEntry
	:members: proc

Note that dbfunctions.TaskEntry does not inherit from db.tracker.models.TaskEntry because inhertinging from Django models is a little more involved than inheriting from a normal python class. Instead, dbfunctions.TaskEntry is a wrapper for db.tracker.models.TaskEntry. 

Instantiating a dbfunctions.TaskEntry object with a database ID number creates an object with basic attributes, e.g., names of linked HDF/plexon files. However, we will often want to perform analyses that are only meaningful for the particular task of a block. For this, we create task-specific child classes, one for each task. These are declared in the semi-misnamed analysis.performance module. 

To get one of these task-specific TaskEntry classes, you can do

.. code-block:: python

	from bmi3d import analysis
	te = analysis.get(id)


Using multiple databases
========================
Django provides options for storing database data using sqlite3, postgres, or mysql. Experiment metadata is best stored using the sqlite3 form, as that ensures that the 
metadata is stored in a file which can be moved easily to multiple machines. The more complex postgres and mysql are not necessary as only one "user" should ever write
to the database at a time. Analysis purposes should only involve reading the databases. 

Django also makes it possible to use multiple databases: https://docs.djangoproject.com/en/dev/topics/db/multi-db/.
To use multiple databases with the dbfn.TaskEntry class, the file $BMI3D/db/settings.py must be told about the other database files. For example, 

.. code-block:: python

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

specifies three different databases which can be used (the 'default' database is the same as the 'bmi3d' database). Different databases of course
must have unique names. Each name specifies a path to a .sql file (note that in each sub-dictionary, a different sqlite3 database is specified). 

After settings.py is modified to include all of your database files, the package will need to be "reconfigured" to know about the new databases.
Reconfiguration can be performed by running the script $BMI3D/make_config.py, which will (re)generate $BMI3D/config. 

To continue using the dbfn.TaskEntry infrastructure, 

.. code-block:: python

    from bmi3d import analysis
    te = analysis.get(id, dbname=DATABASE_NAME)

where DATABASE_NAME is of one of the possible databases you listed in settings.py. If this keyword argument is not specified, dbfn.TaskEntry will use the 'default' database.

basicanalysis
=============
..  automodule:: analysis.basicanalysis
    :members:

Performance metrics
-------------------
For cursor tasks, there are various classes of performance metric:

block-level performance
    hold errors per minute, per successful trial
    timeout penalties per minute

trial-level performance
    movement error
    reach time
    path length
    integrated distance to target

``instantaneous'' performance
    angular error
