import json
import cPickle
import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from riglib import experiment
from tasks import tasklist

class ExpRun(object):
    def __init__(self):
        self.state = None
        self.experiment = None
        self.expidx = None
        self.db = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/")
    
    def make_params(self, task, feats, data):
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
    
    def run(self, task, feats, seq, params, saveid):
        Exp = experiment.make(tasklist[task], feats=feats)
        seqid, gen, args = self._sequence(seq)
        gen = gen(Exp, **args)
        
        if saveid is not None:
            self.expidx = saveid
            db = self.db
            class CommitFeat(object):
                def _start_None(self):
                    super(CommitFeat, self)._start_None()
                    db.save_log(saveid, self.event_log)
            Exp = experiment.make(tasklist[task], feats=[CommitFeat]+feats)
        
        self.experiment = Exp(gen, **cPickle.loads(params))
        self.experiment.start()
        self.state = ("testing", "running")[saveid is not None]
        return params
    
    def _sequence(self, seq):
        if "seq" in seq:
            return seq['id'], experiment.generate.runseq, dict(seq=seq['seq'])
        elif seq['static']:
            gen = experiment.genlist[seq['gen']](**seq['params'])
            return seq['id'], experiment.generate.runseq, dict(seq=gen)
        else:
            return seq['id'], experiment.genlist[seq['gen']], seq['params']
    
    def stop(self):
        self.experiment.end_task()
        state = self.state
        self.experiment = None
        self.state = None
        self.expidx = None
        return state
    
    def pause(self):
        if self.state == "running":
            self.experiment.pause = True
            self.state = "pause"
        elif self.state == "pause":
            self.experiment.pause = False
            self.state = "running"

    def get_status(self):
        return self.state
    
    def get_expidx(self):
        return self.expidx
    
    def report(self):
        return experiment.report(self.experiment)

class RequestHandler(SimpleXMLRPCRequestHandler):
    pass

def run(status=None):
    server = SimpleXMLRPCServer(("localhost", 8001), requestHandler=RequestHandler, allow_none=True)
    server.register_introspection_functions()
    server.register_instance(ExpRun())

    while status is None or status.value == 1:
        try:
            server.handle_request()
        except KeyboardInterrupt:
            status.value = 0

if __name__ == "__main__":
    run()