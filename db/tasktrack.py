import json
import cPickle
import xmlrpclib
import multiprocessing as mp
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from riglib import experiment
from tasks import tasklist

def make_params(task, feats, data):
    Exp = experiment.make(tasklist[task], feats=feats)
    traits = Exp.class_traits()
    params = dict()
    for k, v in data.items():
        v = json.loads(v)
        ttype = traits[k].trait_type.__class__
        #map the user input (always a decoded json object) into a type understood by traits
        if ttype in experiment.typemap:
            v = experiment.typemap[ttype](v)
        params[k] = v
    print params
    return cPickle.dumps(params)

class Tracker(object):
    def __init__(self):
        self.state = None
        self.task = None
        self.proc = None

        ("testing", "running")[saveid is not None]

    def __getattr__(self, attr):
        try
            return super(Tracker, self).__getattr__(attr)
        except:
            self.task.__getattr__(attr)

    def start(self, *args, saveid=None):
        self.proc = mp.Process(target=Task, args=args)
        self.task = xmlrpclib.ServerProxy("http://localhost:8001/", allow_none=True)

    def stop(self):
        self.task.end_task()
        state = self.state
        self.task = None
        self.state = None
        return state
    
    def pause(self):
        if self.state == "running":
            self.task.pause = True
            self.state = "pause"
        elif self.state == "pause":
            self.task.pause = False
            self.state = "running"


def _sequence(self, seq):
    if "seq" in seq:
        return seq['id'], experiment.generate.runseq, dict(seq=seq['seq'])
    elif seq['static']:
        gen = experiment.genlist[seq['gen']](**seq['params'])
        return seq['id'], experiment.generate.runseq, dict(seq=gen)
    else:
        return seq['id'], experiment.genlist[seq['gen']], seq['params']

database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/")
class Task(object):
    def __init__(self, task, feats, seq, params, saveid=None):
        Exp = experiment.make(tasklist[task], feats=feats)
        seqid, gen, args = _sequence(seq)
        gen = gen(Exp, **args)
        
        if saveid is not None:
            class CommitFeat(object):
                def _start_None(self):
                    super(CommitFeat, self)._start_None()
                    database.save_log(saveid, self.event_log)
            Exp = experiment.make(tasklist[task], feats=[CommitFeat]+feats)
        
        exp = Exp(gen, **cPickle.loads(params))
        exp.start()
        self.task = task

    def report(self):
        return experiment.report(self.task)

def runtask(*args):
    task = Task(*args)
    server = SimpleXMLRPCServer(("localhost", 8001), requestHandler=RequestHandler, allow_none=True)
    server.register_introspection_functions()
    server.register_instance(task)
    while task