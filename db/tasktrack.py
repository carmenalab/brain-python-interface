import __builtin__
import os
import ast
import json
import cPickle
import xmlrpclib
import multiprocessing as mp
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

import numpy as np

from riglib import experiment
from tasks import tasklist
from tracker import models

def json_params(params):
    #Fucking retarded ass json implementation in python is retarded as SHIT
    #It doesn't let you override the default encoders! I have to pre-decode 
    #the goddamned object before I push it through json

    def encode(obj):
        if isinstance(obj, models.models.Model):
            return dict(
                __django_model__=obj.__class__.__name__,
                pk=obj.pk)
        elif isinstance(obj, tuple):
            return dict(__builtin__="tuple", args=[obj])
        elif isinstance(obj, np.ndarray):
            return obj.tolist()    
        elif isinstance(obj, dict):
            return dict((k, encode(v)) for k, v in obj.items())
        else:
            return obj
    
    return json.dumps(encode(params))

def param_objhook(obj):
    if '__django_model__' in obj:
        model = getattr(models, obj['__django_model__'])
        return model(pk = obj['pk'])
    elif '__builtin__' in obj:
        func = getattr(__builtin__, obj['__builtin__'])
        return func(*obj['args'])
    return obj

def norm_trait(trait, value):
    ttype = trait.trait_type.__class__.__name__
    if ttype == 'Instance':
        if isinstance(value, int):
            #we got a primary key, lookup class name
            cname = trait.trait_type.klass
            if isinstance(cname, str):
                #We got a class name, it's probably model.Something
                #Split it, then get the model from the models module
                cname = getattr(models, cname.split('.')[-1])
            #otherwise, the klass is actually the class already, and we can directly instantiate

            value = cname.objects.get(pk=value)
        #Otherwise, let's hope it's already an instance
    elif ttype == 'Tuple':
        #Let's make sure this works, for older batches of data
        value = tuple(value)
        
    #use Cast to validate the value
    return trait.cast(value)

def norm_params(task, feats, params):
    Exp = experiment.make(task, feats=feats)
    traits = Exp.class_traits()
    processed = dict()
    
    for name, value in params.items():
        if isinstance(value, (str, unicode)):
            try:
                #assume we got a sanitized value from the database
                #also works for json-valid inputs
                value = json.loads(value, object_hook=param_objhook)
            except:
                #Or try to parse python syntax
                value = ast.literal_eval(value)
        
        #Pushing the value through cast ensures that the value is valid
        processed[name] = norm_trait(traits[name], value)
    
    return json_params(processed)

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
        try:
            print self.task.get_state()
        except:
            pass
        state = self.state
        self.task = None
        self.state = None
        return state

def _sequence(taskid, data, save=True):
    if isinstance(data, dict):
        params = dict()
        print data['params']
        for n, p in data['params'].items():
            try:
                params[n] = json.loads(p)
            except:
                print p
                params[n] = ast.literal_eval(p)
        
        seqdb = models.Sequence(generator_id=data['generator'], 
            task_id=taskid, name=data['name'], 
            params=json.dumps(params))
        if data['static']:
            seqdb.sequence = seqdb.generator.get()(gen(**params))
        
        if save:
            seqdb.save()
    else:
        seqdb = models.Sequence.objects.get(pk=data)
    
    return seqdb


database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
class Task(object):
    def __init__(self, task, feats, seq, params, saveid=None):
        Exp = experiment.make(task.get(), feats=feats)
        gen, gp = seq.get()
        sequence = gen(Exp, **gp)
        
        if saveid is not None:
            class CommitFeat(object):
                def _start_None(self):
                    super(CommitFeat, self)._start_None()
                    database.save_log(saveid, self.event_log)
            Exp = experiment.make(tasklist[task], feats=[CommitFeat]+feats)
        
        exp = Exp(sequence, **json.loads(params, object_hook=param_objhook))
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
            pass
    
    task = Task(*args)
    server.register_instance(task)
    
    while status.value == 1 and task.task.state is not None:
        try:
            server.handle_request()
        except KeyboardInterrupt:
            status.value = 0
    
    del server
    print "exited"