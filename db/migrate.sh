#!/bin/bash
# A sequence of commands to migrate a database created in the python 2 
# version of the software to the python 3 version
cp tracker/models_old.py tracker/models.py
python migrate_db.py
cp tracker/models_new.py tracker/models.py
python manage.py migrate --fake-initial
