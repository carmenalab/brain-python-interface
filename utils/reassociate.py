import re
import os
import json
import glob
from db.tracker import models

plexon = models.System.objects.get(name='plexon')
plex2 = models.System.objects.get(name='plexon2')
hdf = models.System.objects.get(name='hdf')

def reassoc_20130106():
    rplex = models.Feature.objects.get(name='relay_plexon')
    rplexb = models.Feature.objects.get(name='relay_plexbyte')

    for te in models.TaskEntry.objects.filter(feats__in=[rplex, rplexb]):
        plxdata = models.DataFile.objects.filter(system=plexon, entry=te)
        if len(plxdata) == 0:
            fname = te.plexfile()
            hdfdata = models.DataFile.objects.filter(system=hdf, entry=te)
            if fname is not None and len(hdfdata) == 1:
               path, ext = os.path.splitext(hdfdata[0].path)
               newname = os.path.join(plexon.path, path)+".plx"
               #os.rename(fname, newname)
               #models.DataFile(system=plexon, entry=te, path=path+'.plx').save()
               print(models.DataFile(system=plexon, entry=te, path=newname))

def move_fs_orphans(path='/storage/plexon/', dest='/storage/plexon/orphans'):
    valid = set(f.get_path() for f in models.DataFile.objects.filter(system=plexon))
    for f in glob.glob(path+"*.plx"):
        if f not in valid:
            fname = os.path.split(f)[1]
            os.rename(f, os.path.join(dest, fname))

def move_archive_orphans(path='/backup/bmi3d/plexon/', dest='/backup/bmi3d/plexon/orphans'):
    query = models.DataFile.objects.filter(system=plexon)
    files = set(os.path.join(path, df.path) for df in query)
    for plx in glob.glob(path+"*.plx"):
        if plx not in files:
            newname = os.path.join(dest, os.path.split(plx)[1])
            os.rename(plx, newname)
            print(newname)

def assoc_plexon2():
    rplex = models.Feature.objects.get(name='relay_plexon')
    rplexb = models.Feature.objects.get(name='relay_plexbyte')

    for te in models.TaskEntry.objects.filter(feats__in=[rplex, rplexb]):
        fname = te.plexfile("/storage/plexon2", True)
        if fname is not None:
            query = models.DataFile.objects.filter(system=hdf, entry=te)
            if len(query) == 0:
                raise Exception('invalid entry, unknown!')

            newname = os.path.splitext(query[0].path)[0]+'.plx'
            os.rename(fname, os.path.join("/storage/plexon2", newname))
            models.DataFile(system=plex2, entry=te, path=newname).save()
            print(newname)

def move_p2_orphans(path='/storage/plexon2/', dest='/storage/plexon2/orphans'):
    valid = set(f.get_path() for f in models.DataFile.objects.filter(system=plex2))
    for f in glob.glob(path+"*.plx"):
        if f not in valid:
            fname = os.path.split(f)[1]
            os.rename(f, os.path.join(dest, fname))

def set_archived():
    for df in models.DataFile.objects.all():
        if not os.path.isfile(df.get_path()):
            df.archived = True
            try:
                df.get_path()
                df.save()
            except IOError:
                print(df.path, "File not found!")

def old():
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
            print("renaming %s to %s"%(os.path.join(path, orphan), os.path.join(path, newname)))
            df = models.DataFile(local=False, path=newname, system=system, entry=target)
            os.rename(os.path.join(path, orphan), os.path.join(path, newname))
            df.save()
            print(df)
