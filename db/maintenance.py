import os
import sys
import shutil
import datetime

os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'

from db.tracker import models

plexon = models.System.objects.get(name='plexon')

def archive(age=60):
    date = datetime.datetime.now() - datetime.timedelta(days=age)
    for df in models.DataFile.objects.filter(system=plexon):
        if df.entry.date < date and not df.archived:
            try:
                print(df.get_path(True))
            except IOError:
                pass

if __name__ == "__main__":
    archive()