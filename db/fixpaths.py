import os
import datetime
from tracker import models

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
    print "Renaming %s to %s"%(datafile.path, newname)