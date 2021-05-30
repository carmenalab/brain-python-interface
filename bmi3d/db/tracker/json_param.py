'''
This module contains functions which convert parameter sets stored
as JSON blobs into python dictionaries or vice versa.
'''
import builtins
import ast
import json
import numpy as np
import os

log_path = os.path.join(os.path.dirname(__file__), '../../log')


def param_objhook(obj):
    '''
    A custom JSON "decoder" which can recognize certain types of serialized python objects
    (django models, function calls, object constructors) and re-create the objects

    Parameters
    ----------
    obj : dict
        The deserialized JSON data in python dictionary form (after calling json.loads)

    Returns
    -------
    object
        If possible, a python object based on the JSON data is created. If not, the original dictionary
        is simply returned.

    '''
    from . import models
    if '__django_model__' in obj:
        model = getattr(models, obj['__django_model__'])
        return model(pk = obj['pk'])
    elif '__builtin__' in obj:
        func = getattr(builtins, obj['__builtin__'])
        return func(*obj['args'])
    elif '__class__' in obj:
        # look up the module
        mod = builtins.__import__(obj['__module__'], fromlist=[obj['__class__']])

        # get the class with the 'getattr' and then run the class constructor on the class data
        return getattr(mod, obj['__class__'])(obj['__dict__'])
    else: # the type of object is unknown, just return the original dictionary
        return obj

def norm_trait(trait, value):
    '''
    Take user input and convert to the type of the trait.
    For example, a user might select a decoder's name/id but the ID needs to be mapped
    to an object for type checking when the experiment is constructed)

    Parameters
    ----------
    trait : trait object
        trait object declared for the runtime-configurable field.
    value : object
        Value of trait set from user input, to be type-checked

    Returns
    -------
    typecast value of trait
    '''
    ttype = trait.trait_type.__class__.__name__
    if ttype == 'Instance':
        # if the trait is an 'Instance' type and the value is a number, then the number gets interpreted as the primary key to a model in the database
        if isinstance(value, int):
            cname = namelist.instance_to_model[trait.trait_type.klass]
            record = cname.objects.get(pk=value)
            value = record.get()
        # Otherwise, let's hope it's already an instance
    elif ttype == 'InstanceFromDB':
        if isinstance(value, int):
            # look up the model name in the trait
            mdl_name = trait.bmi3d_db_model
            # get the database Model class from 'db.tracker.models'
            with open(os.path.join(log_path, "json_param_log"), "w") as f:
                f.write(str(trait) + "\n")
                f.write(str(mdl_name) + "\n")

            from . import models
            Model = getattr(models, mdl_name)
            record = Model.objects.get(pk=value)
            value = record.get()
        # Otherwise, let's hope it's already an instance
    elif ttype == 'DataFile':
        # Similar to Instance traits, except we always know to use models.DataFile as the database table to look up the primary key
        from . import models
        if isinstance(value, int):
            record = models.DataFile.objects.get(pk=value)
            value = record.get()
    elif ttype == 'Bool':
        if value == 'on':
            value = True
        elif value == 'off':
            value = False
        elif value == 'true':
            value = True
        elif value == 'false':
            value = False
    elif ttype == 'Tuple':
        # Explicit cast to tuple for backwards compatibility reasons (should not be necessary for newer versions of the code/traits lib?)
        value = tuple(value)
    else:
        if isinstance(value, str):
            value = _parse_str(value)

    #use Cast to validate the value
    try:
        if trait.cast is not None:
            return trait.cast
        else:
            # this interface changed in later versions of the trait library
            return trait.validate(trait, 'casting', value)
    except:
        f = open(os.path.join(log_path, "trait_log"), 'w')
        f.write('Error with type for trait %s, %s, value %s' % (str(trait), str(ttype), str(value)))
        f.close()
        import traceback
        traceback.print_exc()
        raise ValueError("Invalid input for parameter %s: %s" % (str(trait.name), str(value)))


def _parse_str(value):
    try:
        return json.loads(value, object_hook=param_objhook)
    except:
        try:
            return ast.literal_eval(value)
        except:
            return value;

class Parameters(object):
    def __init__(self, rawtext):
        if rawtext == '':
            rawtext = '{}'
        self.params = json.loads(rawtext, object_hook=param_objhook)

    @classmethod
    def from_dict(cls, params):
        c = cls('null')
        c.params = params
        return c

    @classmethod
    def from_html(cls, params):
        processed = dict()
        for name, value in list(params.items()):
            if isinstance(value, str):
                processed[name] = _parse_str(value)
            elif isinstance(value, list):
                processed[name] = list(map(_parse_str, value))
            else:
                processed[name] = value

        return cls.from_dict(processed)

    def to_json(self):
        from django.db import models
        def encode(obj):
            if isinstance(obj, models.Model):
                # If an object is a Django model instance, serialize it using just the model name and the primary key
                return dict(
                    __django_model__=obj.__class__.__name__,
                    pk=obj.pk)
            elif isinstance(obj, tuple):
                # for some reason, the re-constructor needs to specified as a tuple?
                return dict(__builtin__="tuple", args=[obj])
            elif isinstance(obj, np.ndarray):
                # serialize numpy arrays as lists
                return obj.tolist()
            elif isinstance(obj, dict):
                # if the object is a dictionary, just run the encoder on each of the 'values' of the dictionary
                return dict((k, encode(v)) for k, v in list(obj.items()))
            elif isinstance(obj, object) and hasattr(obj, '__dict__'):
                # if the object is a new-style class (inherits from 'object'), save the module, class name and object data
                # (python object data (attributes) are stored in the parameter __dict__)
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
        '''
        Apply typecasting to parameters which correspond to experiment traits

        Parameters
        ----------
        traits : dict
            keys are the names of each trait for the Experiment class, values are the trait objects

        Returns
        -------
        None
        '''
        params = self.params
        self.params = dict()
        for name, value in list(params.items()):
            if name in traits:
                traits[name].name = name
                self.params[name] = norm_trait(traits[name], value)
            else:
                self.params[name] = value

    def get_data(self):
        """ Return data with values converted to numbers if possible """
        params_parsed = dict()
        for key in self.params:
            # The internet suggests this might be the best way to check
            # if a string is a number...
            try:
                params_parsed[key] = float(self.params[key])
            except:
                params_parsed[key] = self.params[key]
        return params_parsed

    def __contains__(self, attr):
        return attr in self.params

    def __getitem__(self, attr):
        return self.params[attr]

    def __setitem__(self, attr, val):
        self.params[attr] = val
