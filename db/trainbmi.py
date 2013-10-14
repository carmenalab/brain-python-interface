'''
Functions to train decoders and store into the database
'''

import re
import cPickle
import tempfile
import xmlrpclib

import namelist
from tracker import models
from plexon import plexfile

cellname = re.compile(r'(\d{1,3})\s*(\w{1})')

from celery import task, chain

@task()
def cache_plx(plxfile):
    """Create cache for plexon file"""
    plx = plexfile.openFile(str(plxfile)) 

@task()
def make_bmi(name, clsname, entry, cells, binlen, tslice):
    """Train BMI

    (see doc for cache_and_train for input argument info)
    """
    print "Beginning decoder training"
    print entry
    print type(entry)
    cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(cells)]

    database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)

    datafiles = models.DataFile.objects.filter(entry_id=entry)
    inputdata = dict((d.system.name, d.get_path()) for d in datafiles)
    decoder = namelist.bmis[clsname](cells=cells, binlen=binlen, tslice=tslice, **inputdata)
    tf = tempfile.NamedTemporaryFile('wb')
    cPickle.dump(decoder, tf, 2)
    tf.flush()
    database.save_bmi(name, int(entry), tf.name)

def cache_and_train(name, clsname, entry, cells, binlen, tslice):
    """Cache plexon file and train BMI

    Parameters
    ----------
    clsname : string
        BMI algorithm name (passed to namelist lookup table 'bmis')
    entry : models.TaskEntry
        Django record of training task
    cells : string
        Single string containing all the units to be in decoder, matching
        format in global regex 'cellname'
    binlen : float
        Time of spike history to consider
    tslice : slice
        Task time to use when training the decoder
    """
    plexon = models.System.objects.get(name='plexon')
    plxfile = models.DataFile.objects.get(system=plexon, entry=entry)

    if not plxfile.has_cache():
        cache = cache_plx.si(plxfile.get_path())
        train = make_bmi.si(name, clsname, entry, cells, binlen, tslice)
        chain(cache, train)()
    else:
        make_bmi.delay(name, clsname, entry, cells, binlen, tslice)
