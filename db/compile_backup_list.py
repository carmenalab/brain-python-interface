'''
Compile a list of all the files which should be backed up
'''
import os
from db import dbfunctions as dbfn
from db.tracker import models

backed_up_tes = list(models.TaskEntry.objects.filter(backup=True))

systems_to_back_up = ['hdf', 'plexon']

datafiles = []
for k,te in enumerate(backed_up_tes):
    try:
        te = dbfn.TaskEntry(te)
        te_datafiles = te.datafiles
        for sysname in systems_to_back_up:
            if sysname in te_datafiles:
                datafiles.append(te_datafiles[sysname])

        if k % 100 == 0:
            print k

        if not te.decoder_record is None:
            datafiles.append(te.decoder_filename)
    except:
        pass

# don't back up video files
datafiles = filter(lambda x: not (x[-4:] == '.avi'), datafiles)

f = open(os.path.expandvars('$HOME/files_to_backup'), 'w')
for datafile in datafiles:
    rel_datafile = os.path.relpath(datafile, '/storage')
    f.write("%s\n" % rel_datafile)
f.close()
