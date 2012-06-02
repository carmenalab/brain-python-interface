'''This module contains functions which help deal with parameter sets stored in the database
as JSON blobs. It processes the values into reasonable dictionaries'''
import __builtin__
import ast
import json
import numpy as np

from tracker import models
from riglib import calibrations

def param_objhook(obj):
    if '__django_model__' in obj:
        model = getattr(models, obj['__django_model__'])
        return model(pk = obj['pk'])
    elif '__builtin__' in obj:
        func = getattr(__builtin__, obj['__builtin__'])
        return func(*obj['args'])
    elif '__class__' in obj:
        mod = __builtin__.__import__(obj['__module__'], fromlist=[obj['__class__']])
        return getattr(mod, obj['__class__'])(obj['__dict__'])
    return obj

def norm_trait(trait, value):
    ttype = trait.trait_type.__class__.__name__
    if ttype == 'Instance':
        if isinstance(value, int):
            #we got a primary key, lookup class name
            cname = trait.trait_type.klass
            if not issubclass(cname, models.models.Model):
                for i, m in instance_to_model.items():
                    if issubclass(cname, i):
                        cname = m
            
            value = cname.objects.get(pk=value)
        #Otherwise, let's hope it's already an instance
    elif ttype == 'Tuple':
        #Let's make sure this works, for older batches of data
        value = tuple(value)
        
    #use Cast to validate the value
    return trait.cast(value)

class Parameters(object):
    def __init__(self, rawtext):
        self.params = json.loads(rawtext, object_hook=param_objhook)
    
    @classmethod
    def from_dict(cls, params):
        c = cls('null')
        c.params = params
        return c
    
    @classmethod
    def from_html(cls, params):
        processed = dict()
        for name, value in params.items():
            if isinstance(value, (str, unicode)):
                try:
                    processed[name] = json.loads(value, object_hook=param_objhook)
                except:
                    try:
                        processed[name] = ast.literal_eval(value)
                    except:
                        processed[name] = value;
            else:
                processed[name] = value

        return cls.from_dict(processed)

    def to_json(self):
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
            elif isinstance(obj, object) and hasattr(obj, '__dict__'):
                data = dict(
                    __module__=obj.__class__.__module__,
                    __class__=obj.__class__.__name__, 
                    __dict__=obj.__dict__)
                if hasattr(obj, '__getstate__'):
                    data['__dict__'] = obj.__getstate__()
                return data
            else:
                return obj
        
        return json.dumps(encode(self.params))
    
    def trait_norm(self, traits):
        params = self.params
        self.params = dict()
        for name, value in params.items():
            self.params[name] = norm_trait(traits[name], value)
    
    def __contains__(self, attr):
        return attr in self.params
    
    def __getitem__(self, attr):
        return self.params[attr]