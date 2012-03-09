'''This module contains functions which help deal with parameter sets stored in the database
as JSON blobs. It processes the values into reasonable dictionaries'''
import __builtin__
import ast
import json
import numpy as np
from django.db import models

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

class Parameters(object):
    def __init__(self, rawtext):
        self.params = json.loads(rawtext, object_hook=param_objhook)
    
    @classmethod
    def from_html(cls, params):
        processed = dict()
        for name, value in params.items():
            try:
                processed[name] = json.loads(value, object_hook=param_objhook)
            except:
                processed[name] = ast.literal_eval(value)
        c = cls('null')
        c.params = processed
        return c

    def to_json(self):
        #Fucking retarded ass json implementation in python is retarded as SHIT
        #It doesn't let you override the default encoders! I have to pre-decode 
        #the goddamned object before I push it through json

        def encode(obj):
            if isinstance(obj, models.Model):
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