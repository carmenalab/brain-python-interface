import os
import glob
import json
import datetime
from tracker import models

def run(simulate=True):
    for datafile in models.DataFile.objects.all():
        suffix = dict(eyetracker="edf", hdf="hdf", plexon="plx")
        date = datafile.entry.date
        thatday = datetime.date(date.year, date.month, date.day)
        nextday = thatday + datetime.timedelta(days=1)
        query = models.TaskEntry.objects.filter(date__gte=thatday, date__lte=nextday)
        entrynums = dict([(e, t) for t, e in enumerate(query.order_by("date"))])
        newname = "{subj}{time}_{num:02}.{suff}".format(
            subj=datafile.entry.subject.name[:4].lower(),
            time="%04d%02d%02d"%(thatday.year, thatday.month, thatday.day),
            num=entrynums[datafile.entry], suff=suffix[datafile.system.name],
            )

        if not os.path.exists(datafile.path):
            print "Removing datafile: %s"%datafile.path
            if not simulate:
                datafile.remove()
        elif datafile.system.name == "plexon":
            last = json.loads(datafile.entry.report)[-1][2]
            files = glob.glob("/storage/plexon/*.plx")
            files = sorted(files, key=lambda f: abs(os.stat(f).st_mtime - last))
            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - last
                print "Plexon file %f tdiff"%tdiff
                if tdiff < 60:
                    if datafile.path != files[0]:
                        print "OH GOD WHAT"
                    else:
                        print "Found plexon file, we're ok: ",
                        print "associating %s to %s"%(files[0], os.path.join(datafile.system.path, newname))
                        if not simulate:
                            os.rename(files[0], os.path.join(datafile.system.path, newname))
                            datafile.path = newname
                            datafile.save()
                else:
                    print "Bad plexon file found..."
                    if not simulate:
                        datafile.remove()
        else:
            print "Renaming %s to %s"%(datafile.path, os.path.join(datafile.system.path, newname))
            if not simulate:
                os.rename(datafile.path, os.path.join(datafile.system.path, newname))
                datafile.path = newname
                datafile.save()