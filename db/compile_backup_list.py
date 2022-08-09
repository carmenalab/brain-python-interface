'''
Compile a list of all the files which should be backed up
'''
import os
from . import dbfunctions as dbfn
from .tracker import models
import datetime

#Get backup list:
backed_up_tes = list(models.TaskEntry.objects.filter(backup=True, date__gte=datetime.datetime(2015, 11, 0o1)))

#Get list of Task Entries already added to list: 
try:
    te_added = open(os.path.expandvars('$HOME/files_to_backup_te_list'), 'r')
    items = te_added.read().split('\n')
    already_te_added = [int(i) for i in items if len(i) > 0]
    te_added.close()
except:
    print('No previous backup list found, making new one')
    te_added = open(os.path.expandvars('$HOME/files_to_backup_te_list'), 'w')
    te_added.close()
    already_te_added = []

systems_to_back_up = ['hdf', 'plexon']

#Open file for writing 
#Open new file: 
f = open(os.path.expandvars('$HOME/files_to_backup'), 'w')

#Append to previous file: 
f_list = open(os.path.expandvars('$HOME/files_to_backup_te_list'), 'a')

for k,te in enumerate(backed_up_tes):

    #Check if already backed up: 
    if te.id in already_te_added:
        pass
    else:
        #Array for datafile names:
        datafiles = []
        try:
            te = dbfn.TaskEntry(te)
            te_datafiles = te.datafiles
            for sysname in systems_to_back_up:
                if sysname in te_datafiles:
                    datafiles.append(te_datafiles[sysname])

            #Close HDF file to free memory: 
            try:
                te.hdf.close()
            except:
                pass


            if k % 100 == 0:
                print(k)

            if not te.decoder_record is None:
                datafiles.append(te.decoder_filename)
        except:
            pass

        #Write to file: 
        if len(datafiles) > 0:
            for datafile in datafiles:
                rel_datafile = os.path.relpath(datafile, '/storage')
                f.write("%s\n" % rel_datafile)
        #Write to 'already backed up file:'
        f_list.write("%s\n" % str(te.id))

f.close()
f_list.close()

# # don't back up video files
# datafiles = filter(lambda x: not (x[-4:] == '.avi'), datafiles)

# for datafile in datafiles:
#     rel_datafile = os.path.relpath(datafile, '/storage')
#     f.write("%s\n" % rel_datafile)
# f.close()
