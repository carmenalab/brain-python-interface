import re
import os
import json
from tracker import models
path = "/storage/plexon/"
plxform = re.compile('cart(\d{2})(\d{2})(\d{4})(\d{3}).plx')

orphans = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and plxform.match(f)]

system = models.System.objects.get(name="plexon")
for orphan in orphans:
    mtime = os.stat(os.path.join(path, orphan)).st_mtime
    day, month, year, num = plxform.match(orphan).groups()
    entries = models.TaskEntry.objects.filter(date__year=year, date__month=month, date__day=day)
    etime = dict([(e, i) for i, e in enumerate(sorted(entries, key=lambda x:x.date))])
    tdiffs = [(e, abs(mtime - json.loads(e.report)[-1][2])) if e.report != '[]' else e for e in entries]
    target, tdiff = sorted(tdiffs, key=lambda x:x[1])[0]
    if tdiff < 60:
        newname = "cart{time}_{num:02}.plx".format(time=target.date.strftime('%Y%m%d'), num=etime[target]+1)
        print "renaming %s to %s"%(os.path.join(path, orphan), os.path.join(path, newname))
        df = models.DataFile(local=False, path=newname, system=system, entry=target)
        os.rename(os.path.join(path, orphan), os.path.join(path, newname))
        df.save()
        print df
