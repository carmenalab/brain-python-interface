import os
import glob
import json
import datetime
from db.tracker import models

def run(simulate=True):
    moved = set()
    for datafile in models.DataFile.objects.all():
        print(datafile)
        suffix = dict(eyetracker="edf", hdf="hdf", plexon="plx")
        date = datafile.entry.date
        thatday = datetime.date(date.year, date.month, date.day)
        nextday = thatday + datetime.timedelta(days=1)
        query = models.TaskEntry.objects.filter(date__gte=thatday, date__lte=nextday)
        entrynums = dict([(e, t) for t, e in enumerate(query.order_by("date"))])
        newname = "{subj}{time}_{num:02}.{suff}".format(
            subj=datafile.entry.subject.name[:4].lower(),
            time="%04d%02d%02d"%(thatday.year, thatday.month, thatday.day),
            num=entrynums[datafile.entry]+1, suff=suffix[datafile.system.name],
            )
        newpath = os.path.join(datafile.system.path, newname)
        oldpath = os.path.join(datafile.system.path, datafile.path)
        
        if not os.path.exists(oldpath):
            print("\tRemoving datafile: %s"%oldpath)
            if not simulate:
                datafile.remove()
        elif oldpath in moved:
            print("OH GOD moving already missing file!!!")
        elif datafile.system.name == "plexon":
            last = json.loads(datafile.entry.report)[-1][2]
            files = glob.glob("/storage/plexon/*.plx")
            files = sorted(files, key=lambda f: abs(os.stat(f).st_mtime - last))
            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - last
                print("\tPlexon file %f tdiff"%tdiff)
                if abs(tdiff) < 60:
                    if oldpath != files[0]:
                        print("\tFound plexon file %s, but recorded %s!"%(files[0], oldpath))
                        print("\tassociating %s to %s"%(files[0], newpath))
                        if not simulate:
                            moved.add(files[0])
                            os.rename(files[0], os.path.join(datafile.system.path, newname))
                            datafile.path = newname
                            datafile.save()
                    else:
                        print("\tPLX file is ok, just renaming %s to %s"%(oldpath, newpath))
                        if not simulate:
                            moved.add(oldpath)
                            os.rename(oldpath, newpath)
                            datafile.path = newname
                            datafile.save()
                else:
                    if oldpath != files[0]:
                        print("\tPlexfile doesn't match best %s vs. %s"%(files[0], oldpath))
                    else:
                        print("\tBad plexon file found, removing %s..."%oldpath)
                        if not simulate:
                            datafile.remove()
        else:
            #print "\tRenaming %s to %s"%(datafile.path, newpath)
            #if not simulate:
            #    moved.add(datafile.path)
            #    os.rename(datafile.path, newpath)
            #    datafile.path = newname
            #    datafile.save()
            pass
