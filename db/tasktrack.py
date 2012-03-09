import os
import json
import time
import cPickle
import xmlrpclib
import multiprocessing as mp
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import numpy as np

from riglib import experiment
from tasks import tasklist
from tracker import models

from json_param import Parameters

class Tracker(object):
    def __init__(self):
        self.state = None
        self.task = None
        self.proc = None
        self.expidx = None
        self.status = mp.Value('b', 1)

    def __getattr__(self, attr):
        try:
            return super(Tracker, self).__getattr__(attr)
        except:
            return self.task.__getattr__(attr)

    def start(self, task, feats, seq, params, saveid=None):
        self.status.value = 1
        self.proc = mp.Process(target=runtask, args=(self.status, task, feats, seq, params, saveid))
        self.proc.start()
        self.task = xmlrpclib.ServerProxy("http://localhost:8001/", allow_none=True)
        self.state = "testing" if saveid is None else "running"
        self.expidx = saveid
    
    def pause(self):
        self.state = self.task.pause()

    def stop(self):
        self.task.end_task()
        state = self.state
        self.state = None
        self.expidx = None
        try:
            print self.task.get_state()
        except:
            print "couldn't close..."
        self.task = None
        return state

database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
class Task(object):
    def __init__(self, task, feats, seq, params, saveid=None):
        if saveid is not None:
            class CommitFeat(object):
                def _start_None(self):
                    super(CommitFeat, self)._start_None()
                    database.save_log(saveid, self.event_log)
            Exp = experiment.make(task.get(), feats=[CommitFeat]+feats)
        else:
            Exp = experiment.make(task.get(), feats=feats)

        gen, gp = seq.get()
        sequence = gen(Exp, **gp.params)
        exp = Exp(sequence, **params)
        exp.start()
        self.task = exp

    def report(self):
        return experiment.report(self.task)
    
    def pause(self):
        self.task.pause = not self.task.pause
        return "pause" if self.task.pause else "running"
    
    def end_task(self):
        self.task.end_task()
    
    def get_state(self):
        return self.task.state

class RequestHandler(SimpleXMLRPCRequestHandler):
    pass

def runtask(status, *args):
    os.nice(0)
    server = None
    while server is None:
        try:
            server = SimpleXMLRPCServer(("localhost", 8001), requestHandler=RequestHandler, allow_none=True)
            server.register_introspection_functions()
        except:
            print "Cannot open server..."
            time.sleep(2.)
    
    task = Task(*args)
    server.register_instance(task)
    server.timeout = 0.5
    
    while status.value == 1 and task.task.state is not None:
        try:
            server.handle_request()
        except KeyboardInterrupt:
            status.value = 0
    
    server.server_close()
    print "exited"


try:
    tracker
except NameError:
    tracker = Tracker()