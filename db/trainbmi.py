import os
import re
import cPickle
import tempfile
import multiprocessing as mp

from namelist import bmis
from tracker import models
from riglib.plexon import plexfile

cellname = re.compile(r'(\d{1,3})\s*(\w{1})')

from celery import task, chain

@task()
def cache_plx(plxfile):
    plx = plexfile.openFile(plxfile) 

@task()
def make_bmi(name, clsname, entry, cells, binlen):
    cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(cells)]

    database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)

    datafiles = models.DataFile.objects.filter(entry_id=entry)
    inputdata = dict((d.system.name, d.get_path()) for d in datafiles)
    decoder = bmis[clsname](cells=cells, binlen=binlen, **inputdata)
    tf = tempfile.NamedTemporaryFile()
    cPickle.dump(tf, decoder, 2)

    database.save_bmi(name, entry.id, tf.name)

def cache_and_train(name, clsname, entry, cells, binlen):
    plexon = models.System.objects.get(name='plexon')
    plxfile = models.DataFile.objects.get(system=plexon, entry=entry)

    if not plxfile.has_cache():
        cache = cache_plx.si(plxfile.get_path())
        train = make_bmi.si(name, clsname, entry, cells, binlen)
        chain(cache, train)()
    else:
        make_bmi.delay(name, clsname, entry, cells, binlen)