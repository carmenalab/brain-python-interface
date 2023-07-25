'''
Classes here which inherit from django.db.models.Model define the structure of the database

Django database modules. See https://docs.djangoproject.com/en/dev/intro/tutorial01/
for a basic introduction
'''

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import json
import pickle
import inspect
import numpy as np
import importlib
import subprocess
import traceback
import imp
import tables
import tempfile
import shutil
from collections import OrderedDict
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

from riglib import calibrations, experiment

def import_by_path(import_path):
    path_components = import_path.split(".")
    module_name = (".").join(path_components[:-1])
    class_name = path_components[-1]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

def func_or_class_to_json(func_or_class, current_values, desc_lookup):
    '''
    Helper for translating a function or class into json parameters for the UI
    '''
    try:
        args = inspect.getargspec(func_or_class)
        names, defaults = args.args, args.defaults
        if "self" in names: names.remove("self")
    except TypeError:
        args = inspect.getargspec(func_or_class.__init__)
        names, defaults = args.args, args.defaults
        names.remove("self")

    params = OrderedDict()
    if len(names) == 0:
        return params
        
    for name, default in zip(names, defaults):
        if name == 'exp':
            continue
        if type(default) is tuple:
            typename = "Tuple"
        elif type(default) is int or type(default) is float:
            typename = "Float"
        elif type(default) is bool:
            typename = "Bool"
        else:
            typename = "String"

        params[name] = dict(type=typename, default=default, desc=desc_lookup(name))
        if name in current_values:
            params[name]['value'] = current_values[name]

    return params

class Task(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(default=True, blank=True)
    import_path = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        if not self.import_path is None and self.import_path != "":
            return "Task[{}]: {}".format(self.name, self.import_path)
        else:
            return "Task[{}]".format(self.name)

    def __repr__(self):
        return self.__str__()

    def get_base_class(self):
        if not self.import_path is None and len(self.import_path) > 0:
            return import_by_path(self.import_path)
        else:
            #print(r"Could not find base class for task %s. No import_path provided" % self.name)
            return experiment.Experiment

    def get(self, feats=(), verbose=False):
        if verbose: print("models.Task.get()")

        feature_classes = Feature.getall(feats)
        task_cls = self.get_base_class()

        if not None in feature_classes:
            try:
                return experiment.make(task_cls, feature_classes)
            except:
                print("Problem making the task class!")
                traceback.print_exc()
                print(self.name)
                print(feats)
                print(Feature.getall(feats))
                print("*******")
                return experiment.Experiment
        else:
            print("Task was not installed properly, defaulting to generic experiment!")
            return experiment.Experiment

    @staticmethod
    def add_new_task(task_name, class_path):
        """ Add new task to the database

        Parameters
        ----------
        task_name : string
            Human-readable name of task.
        class_path : string
            Python import path to the class, for example, riglib.experiment.Experiment
        """
        if " " in class_path:
            raise ValueError("Class path cannot have spaces!")

        try:
            path_components = class_path.split(".")
            module_name = (".").join(path_components[:-1])
            class_name = path_components[-1]
            module = importlib.import_module(module_name)
            task_cls = module.getattr(class_name)
        except:
            raise ImportError("Error importing class")
            traceback.print_exc()
        else:
            # finish adding to table
            Task(name=task_name, import_path=class_path).save()

    def params(self, feats=(), values=None):
        '''
        Get user-editable parameters for the frontend

        Parameters
        ----------
        feats : iterable of Feature instances
            Features selected on the task interface
        values : dict
            Values for the task parameters

        '''
        Exp = self.get(feats=feats)
        params = Exp.get_params()

        if values is None:
            values = dict()

        for trait_name in params:
            if params[trait_name]['type'] in ['InstanceFromDB', 'DataFile']:
                mdl_name, filter_kwargs = params[trait_name]['options']

                # get the database Model class from 'db.tracker.models'
                Model = globals()[mdl_name]

                # look up database records which match the model type & filter parameters
                insts = Model.objects.filter(**filter_kwargs).order_by("-date")
                params[trait_name]['options'] = [(i.pk, i.path) for i in insts]

            if trait_name in values:
                params[trait_name]['value'] = values[trait_name]
        return params
        # #from namelist import instance_to_model, instance_to_model_filter_kwargs


        # # Use an ordered dict so that params actually stay in the order they're added, instead of random (hash) order
        # params = OrderedDict()

        # # Run the meta-class constructor to make the Task class (base task class + features )
        # Exp = self.get(feats=feats)
        # ctraits = Exp.class_traits()

        # def add_trait(trait_name):
        #     trait_params = dict()
        #     trait_params['type'] = ctraits[trait_name].trait_type.__class__.__name__
        #     trait_params['default'] = _get_trait_default(ctraits[trait_name])
        #     trait_params['desc'] = ctraits[trait_name].desc
        #     trait_params['hidden'] = 'hidden' if Exp.is_hidden(trait_name) else 'visible'
        #     if hasattr(ctraits[trait_name], 'label'):
        #         trait_params['label'] = ctraits[trait_name].label
        #     else:
        #         trait_params['label'] = trait_name

        #     if trait_name in values:
        #         trait_params['value'] = values[trait_name]

        #     if trait_params['type'] == "InstanceFromDB":
        #         # look up the model name in the trait
        #         mdl_name = ctraits[trait_name].bmi3d_db_model

        #         # get the database Model class from 'db.tracker.models'
        #         Model = globals()[mdl_name]
        #         filter_kwargs = ctraits[trait_name].bmi3d_query_kwargs

        #         # look up database records which match the model type & filter parameters
        #         insts = Model.objects.filter(**filter_kwargs).order_by("-date")
        #         trait_params['options'] = [(i.pk, i.path) for i in insts]

        #     elif trait_params['type'] == 'Instance':
        #         raise ValueError("You should use the 'InstanceFromDB' trait instead of the 'Instance' trait!")

        #     # if the trait is an enumeration, look in the 'Exp' class for
        #     # the options because for some reason the trait itself can't
        #     # store the available options (at least at the time this was written..)
        #     elif trait_params['type'] == "Enum":
        #         raise ValueError("You should use the 'OptionsList' trait instead of the 'Enum' trait!")

        #     elif trait_params['type'] == "OptionsList":
        #         trait_params['options'] = ctraits[trait_name].bmi3d_input_options

        #     elif trait_params['type'] == "DataFile":
        #         # look up database records which match the model type & filter parameters
        #         filter_kwargs = ctraits[trait_name].bmi3d_query_kwargs
        #         insts = DataFile.objects.filter(**filter_kwargs).order_by("-date")
        #         trait_params['options'] = [(i.pk, i.path) for i in insts]

        #     params[trait_name] = trait_params

        #     if trait_name == 'bmi': # a hack for really old data, where the 'decoder' was mistakenly labeled 'bmi'
        #         params['decoder'] = trait_params

        # # add all the traits that are explicitly instructed to appear at the top of the menu
        # ordered_traits = Exp.ordered_traits
        # for trait in ordered_traits:
        #     if trait in Exp.class_editable_traits():
        #         add_trait(trait)

        # # add all the remaining non-hidden traits
        # for trait in Exp.class_editable_traits():
        #     if trait not in params and not Exp.is_hidden(trait):
        #         add_trait(trait)

        # # add any hidden traits
        # for trait in Exp.class_editable_traits():
        #     if trait not in params:
        #         add_trait(trait)

    def sequences(self):
        from .json_param import Parameters
        seqs = dict()
        for s in Sequence.objects.filter(task=self.id):
            try:
                seqs[s.id] = s.to_json()
            except:
                print('Sequence cannot be Accessed: ', s.id)
        return seqs

    def get_generators(self):
        # Supply sequence generators which are declared to be compatible with the selected task class
        exp_generators = []
        Exp = self.get()
        if hasattr(Exp, 'sequence_generators'):
            for seqgen_name in Exp.sequence_generators:
                try:
                    g = Generator.objects.using(self._state.db).get(name=seqgen_name)
                    exp_generators.append([g.id, seqgen_name])
                except:
                    print("missing generator %s" % seqgen_name)
                    traceback.print_exc()
        return exp_generators

    def controls(self, feats=()):
        exp = self.get(feats=feats)
        ctl = exp.controls
        values = dict()

        def desc_lookup(name):
            table = {
                'msg': 'Metadata to mark this moment in experiment',
            }
            if name in table.keys():
                return table[name]
            else:
                return name

        controls = []
        for c in ctl:
            params = func_or_class_to_json(c, values, desc_lookup)
            if 'static' in params:
                params.pop('static')
                controls.append(dict(name=c.__name__, params=params, static=True))
            else:
                controls.append(dict(name=c.__name__, params=params))

        return controls

def can_be_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

class Feature(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(blank=True, default=True)
    import_path = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return "Feature[{}]".format(self.name)
    def __repr__(self):
        return self.__str__()

    @property
    def desc(self):
        feature_cls = self.get()
        if not feature_cls is None:
            return feature_cls.__doc__
        else:
            return ''

    def get(self, update_builtin=False):
        from features import built_in_features
        if self.import_path is not None:
            return import_by_path(self.import_path)
        elif self.name in built_in_features:
            import_path = built_in_features[self.name].__module__ + '.' + built_in_features[self.name].__qualname__
            if update_builtin:
                self.import_path = import_path
                self.save()
                return import_by_path(self.import_path)
            else:
                #print("Feature %s import path not found, but found a default" % self.name)
                return import_by_path(import_path)
        else:
            print("Feature %s has no import_path" % self.name)
            return None

    @staticmethod
    def getall(feats):
        feature_class_list = []
        for feat in feats:
            if isinstance(feat, Feature):
                feat_cls = feat.get()
            elif isinstance(feat, int):
                feat_cls = Feature.objects.get(pk=feat).get()
            elif isinstance(feat, str) and can_be_int(feat):
                feat_cls = Feature.objects.get(pk=int(feat)).get()
            elif isinstance(feat, str):
                feat_cls = Feature.objects.get(name=feat).get()
            else:
                print("Cannot find feature: ", feat)

            feature_class_list.append(feat_cls)
        return feature_class_list

class System(models.Model):
    """ Representation for systems which generate data (similar to a data type)"""
    name = models.CharField(max_length=128)
    path = models.TextField()
    archive = models.TextField()
    processor_path = models.CharField(max_length=200, blank=True, null=True)
    input_path = models.TextField(blank=True, null=True)

    def __str__(self):
        return "System[{}]".format(self.name)
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def populate():
        for name in ["eyetracker", "hdf", "plexon", "bmi", "bmi_params", "juice_log", "blackrock"]:
            try:
                System.objects.get(name=name)
            except ObjectDoesNotExist:
                System(name=name, path="/storage/rawdata/%s"%name).save()

    def get_post_processor(self):
        if self.processor_path is not None and len(self.processor_path) > 0:
            return import_by_path(self.processor_path)
        else:
            return lambda x: x # identity fn

    @staticmethod
    def make_new_sys(name):
        try:
            new_sys_rec = System.objects.get(name=name)
        except ObjectDoesNotExist:
            data_dir = "/storage/rawdata/%s" % name
            new_sys_rec = System(name=name, path=data_dir)
            new_sys_rec.save()
            os.popen('mkdir -p %s' % data_dir)

        return new_sys_rec

    def save_to_file(self, obj, filename, obj_name=None, entry_id=-1):
        full_filename = os.path.join(self.path, filename)
        pickle.dump(obj, open(full_filename, 'w'))

        if obj_name is None:
            obj_name = filename.rstrip('.pkl')

        df = DataFile()
        df.path = filename
        df.system = self
        df.entry_id = entry_id
        df.save()

class Subject(models.Model):
    name = models.CharField(max_length=128)
    def __str__(self):
        return "Subject[{}]".format(self.name)
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_all_subjects(dbname=None):
        subjects = Subject.objects.using(dbname).all().order_by("name")
        return [s.name for s in subjects]

class Experimenter(models.Model):
    name = models.CharField(max_length=128)
    def __str__(self):
        return "Experimenter[{}]".format(self.name)
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_all_experimenters(dbname=None):
        experimenters = Experimenter.objects.using(dbname).all().order_by("name")
        return [s.name for s in experimenters]

class Generator(models.Model):
    name = models.CharField(max_length=128)
    params = models.TextField()
    static = models.BooleanField()
    visible = models.BooleanField(blank=True, default=True)

    def __str__(self):
        return "Generator[{}]".format(self.name)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def get_all_generators(dbname=None):
        generator_names = []
        generator_functions = []
        tasks = Task.objects.using(dbname).all()
        for task in tasks:
            try:
                task_cls = task.get()
            except:
                # if a task is not importable, then it cannot have any detectable generators
                continue

            if hasattr(task_cls, 'sequence_generators'):
                generator_function_names = task_cls.sequence_generators
                gen_fns = [getattr(task_cls, x) if hasattr(task_cls, x) else None for x in generator_function_names]
                for fn_name, fn in zip(generator_function_names, gen_fns):
                    if fn in generator_functions or fn is None:
                        pass
                    else:
                        generator_names.append(fn_name)
                        generator_functions.append(fn)

        generators = dict()
        for fn_name, fn in zip(generator_names, generator_functions):
            generators[fn_name] = fn
        return generators

    def get(self):
        '''
        Retrieve the function that can be used to construct the ..... generator? sequence?
        '''
        generators = Generator.get_all_generators()
        return generators[self.name]

    @staticmethod
    def remove_unused():
        generators = Generator.get_all_generators()
        listed_generators = set(generators.keys())
        db_generators = set(gen.name for gen in Generator.objects.all())

        # determine which generators are unused in the database using set subtraction
        unused_generators = db_generators - listed_generators
        for name in unused_generators:
            try:
                Generator.objects.filter(name=name).delete()
            except models.ProtectedError:
                pass

    @staticmethod
    def populate():
        generators = Generator.get_all_generators()
        listed_generators = set(generators.keys())
        db_generators = set(gen.name for gen in Generator.objects.all())

        # determine which generators are missing from the database using set subtraction
        missing_generators = listed_generators - db_generators
        for name in missing_generators:

            # TODO not sure why we're populating the 'params' field here since it never gets used    

            # The sequence/generator constructor can either be a callable or a class constructor... not aware of any uses of the class constructor
            try:
                args = inspect.getargspec(generators[name]).args
                print(args)
            except TypeError:
                args = inspect.getargspec(generators[name].__init__).args
                args.remove("self")

            # A generator is determined to be static only if it takes an "exp" argument representing the Experiment class
            static = not ("exp" in args)
            if "exp" in args:
                args.remove("exp")

            # TODO not sure why the 'length' argument is being removed; is it assumed that all generators will take a 'length' argument?
            # if "length" in args:
            #     args.remove("length")

            gen_obj = Generator(name=name, params=",".join(args), static=static)
            gen_obj.save()

    def to_json(self, values=None):
        if values is None:
            values = dict()
        gen = self.get()

        def desc_lookup(name):
            table = {
                'nblocks': 'Number of trials times number of unique targets',
                'ntrials': 'Number of trials',
                'nreps': 'The number of repetitions of each unique condition.',
                'ntargets': 'Number of (evenly spaced) targets',
                'pos': 'Position of the target',
                'distance': 'The distance in cm between the center and peripheral targets',
                'origin': 'Location of the central targets around which the peripheral targets span',
                'boundaries': 'The limits of the allowed target locations',
                'chain_length': 'Number of targets in each sequence before a reward',
            }
            if name in table.keys():
                return table[name]
            else:
                return name

        params = func_or_class_to_json(gen, values, desc_lookup)
        return dict(name=self.name, params=params)

class Sequence(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    generator = models.ForeignKey(Generator, on_delete=models.PROTECT)
    name = models.CharField(max_length=256)
    params = models.TextField() #json data
    sequence = models.TextField(blank=True) #pickle data
    task = models.ForeignKey(Task, on_delete=models.PROTECT)

    def __str__(self):
        return "Sequence[{}] of type Generator[{}]".format(self.name, self.generator.name)

    def __repr__(self):
        return self.__str__()

    def get(self):
        from riglib.experiment import generate
        from .json_param import Parameters

        if hasattr(self, 'generator') and self.generator.static: # If the generator is static, (NOTE: the generator being static is different from the *sequence* being static)
            if len(self.sequence) > 0:
                import ast
                return generate.runseq, dict(seq=pickle.loads(ast.literal_eval(self.sequence)))
            else:
                return generate.runseq, dict(seq=self.generator.get()(**Parameters(self.params).params))
        else:
            return self.generator.get(), Parameters(self.params).params

    def to_json(self):
        from .json_param import Parameters
        state = 'saved' if self.pk is not None else "new"
        js = dict(name=self.name, state=state)
        js['static'] = len(self.sequence) > 0
        js['params'] = self.generator.to_json(Parameters(self.params).params)['params']
        js['generator'] = self.generator.id, self.generator.name
        return js

    @classmethod
    def from_json(cls, js):
        '''
        Construct a models.Sequence instance from JSON data (e.g., generated by the web interface for starting experiments)
        '''
        from .json_param import Parameters

        # Error handling when input argument 'js' actually specifies the primary key of a Sequence object already in the database
        try:
            seq = Sequence.objects.get(pk=int(js))
            print("retreiving sequence from POSTed ID")
            return seq
        except:
            pass

        # Make sure 'js' is a python dictionary
        if not isinstance(js, dict):
            js = json.loads(js)

        # Determine the ID of the "generator" used to make this sequence
        genid = js['generator']
        if isinstance(genid, (tuple, list)):
            genid = genid[0]

        # Construct the database record for the new Sequence object
        seq = cls(generator_id=int(genid), name=js['name'])

        # Link the generator instantiation parameters to the sequence record
        # Parameters are stored in JSON format in the database
        seq.params = Parameters.from_html(js['params']).to_json()

        # If the sequence is to be static,
        if js['static']:
            print("db.tracker.models.Sequence.from_json: storing static sequence data to database")
            generator_params = Parameters(seq.params).params
            seq_data = seq.generator.get()(**generator_params)
            seq.sequence = pickle.dumps(seq_data)
        return seq

class TaskEntry(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.PROTECT)
    experimenter = models.ForeignKey(Experimenter, null=True, on_delete=models.PROTECT)
    date = models.DateTimeField(auto_now_add=True)
    task = models.ForeignKey(Task, on_delete=models.PROTECT)
    feats = models.ManyToManyField(Feature)
    sequence = models.ForeignKey(Sequence, blank=True, null=True, on_delete=models.PROTECT)
    project = models.TextField()
    session = models.TextField()

    params = models.TextField()
    report = models.TextField()
    notes = models.TextField()
    metadata = models.TextField(default="") # serialized custom key/value metadata
    visible = models.BooleanField(blank=True, default=True)
    backup = models.BooleanField(blank=True, default=False)
    template = models.BooleanField(blank=True, default=False)
    entry_name = models.CharField(blank=True, null=True, max_length=50)
    sw_version = models.CharField(blank=True, null=True, max_length=100)

    def __str__(self):
        return "{date}: {subj} on {task} task, id={id}".format(
            date=str(self.date),
            subj=self.subject.name,
            task=self.task.name,
            id=self.id)

    def __repr__(self):
        return self.__str__()

    @property
    def ui_id(self):
        if self.entry_name is not None and self.entry_name != "":
            return "%s (%d)" % (self.entry_name, self.id)
        else:
            return str(self.id)

    def get(self, feats=()):
        from .json_param import Parameters
        from riglib import experiment
        Exp = experiment.make(self.task.get(), tuple(f.get() for f in self.feats.all())+feats)
        params = Parameters(self.params)
        params.trait_norm(Exp.class_traits())
        if issubclass(Exp, experiment.Sequence):
            gen, gp = self.sequence.get()
            seq = gen(Exp, **gp)
            exp = Exp(seq, **params.params)
        else:
            exp = Exp(**params.params)
        exp.event_log = json.loads(self.report)
        return exp

    def update_date(self, *date_args):
        """Utility function to update the date of a record if set improperly (e.g., if entered manually)"""
        import datetime
        if len(date_args) == 1 and isinstance(date_args[0], datetime.datetime):
            self.date = date_args[0]
            self.save()
        elif len(date_args) >= 3 and len(date_args) <= 7: # assume these are numbers with fields split out
            self.date = datetime.datetime(*date_args)
            self.save()
        else:
            raise ValueError("Unrecognized date arguments: ", date_args)

    @property
    def task_params(self):
        from .json_param import Parameters
        data = Parameters(self.params).params

        if 'bmi' in data:
            data['decoder'] = data['bmi']
        ##    del data['bmi']
        return data

    @property
    def task_metadata(self):
        from .json_param import Parameters
        data = Parameters(self.metadata).params
        return data

    @property
    def sequence_params(self):
        from .json_param import Parameters
        data = Parameters(self.sequence.params).params
        return data

    @staticmethod
    def get_default_metadata():
        subject = {
            'type': 'Enum',
            'default': '',
            'desc': 'Who',
            'hidden': 'visible',
            'options':  Subject.get_all_subjects(),
            'required': True
        }
        experimenter = {
            'type': 'Enum',
            'default': '',
            'desc': 'Who is running the experiment',
            'hidden': 'visible',
            'options':  Experimenter.get_all_experimenters(),
            'required': True
        }
        project = {
            'type': 'String',
            'default': '',
            'desc': 'Which project are you working on',
            'hidden': 'visible',
            'value': '',
            'required': True
        }
        session = {
            'type': 'String',
            'default': '',
            'desc': 'Specific instance of the project',
            'hidden': 'visible',
            'value': '',
            'required': True
        }
        metadata = {
            'subject': subject,
            'experimenter': experimenter,
            'project': project,
            'session': session,
        }
        return metadata

    def plexfile(self, path='/storage/plexon/', search=False):
        rplex = Feature.objects.get(name='relay_plexon')
        rplexb = Feature.objects.get(name='relay_plexbyte')
        feats = self.feats.all()
        if rplex not in feats and rplexb not in feats:
            return None

        if not search:
            system = System.objects.get(name='plexon')
            df = DataFile.objects.filter(entry=self.id, system=system)
            if len(df) > 0:
                return df[0].get_path()

        if len(self.report) > 0:
            event_log = json.loads(self.report)
            import os, sys, glob, time
            if len(event_log) < 1:
                return None

            start = event_log[-1][2]
            files = sorted(glob.glob(path+"/*.plx"), key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < 60:
                     return files[0]

    def offline_report(self):
        Exp = self.task.get(self.feats.all())

        if len(self.report) == 0:
            return dict()
        else:
            report = json.loads(self.report)
            if isinstance(report, list):
                # old method: calculate from full event log
                rpt = Exp.offline_report(report)
            else:
                # new method: reformat stats
                rpt = Exp.format_log_summary(report)


            ## If this is a BMI block, add the decoder name to the report (doesn't show up properly in drop-down menu for old blocks)
            # try:
            #     from db import dbfunctions
            #     te = dbfunctions.TaskEntry(self.id, dbname=self._state.db)
            #     rpt['Decoder name'] = te.decoder_record.name + ' (trained in block %d)' % te.decoder_record.entry_id
            # except AttributeError:
            #     pass
            # except:
            #     import traceback
            #     traceback.print_exc()
            return rpt

    def to_json(self):
        '''
        Create a JSON dictionary of the metadata associated with this block for display in the web interface
        '''
        print("starting TaskEntry.to_json()")
        from .json_param import Parameters

        # Run the metaclass constructor for the experiment used. If this can be avoided, it would help to break some of the cross-package software dependencies,
        # making it easier to analyze data without installing software for the entire rig

        Exp = self.task.get(self.feats.all())
        from . import exp_tracker
        tracker = exp_tracker.get()
        if tracker.get_status() and tracker.proc is not None and hasattr(tracker.proc, 'saveid') and tracker.proc.saveid == self.id:
            state = tracker.get_status()#'completed' if self.pk is not None else "new"
        else:
            state = 'completed'  
        js = dict(task=self.task.id, state=state, subject=self.subject.id, notes=self.notes)
        js['feats'] = dict([(f.id, f.name) for f in self.feats.all()])
        js['params'] = self.task.params(self.feats.all(), values=self.task_params)
        js['controls'] = self.task.controls(feats=self.feats.all())

        if len(js['params'])!=len(self.task_params):
            print('param lengths: JS:', len(js['params']), 'Task: ', len(self.task_params))

        # Add metadata
        js['metadata'] = self.get_default_metadata()
        js['metadata']['subject']['default'] = self.subject.name
        if self.experimenter: 
            js['metadata']['experimenter']['default'] = self.experimenter.name
        js['metadata']['project']['value'] = self.project
        js['metadata']['session']['value'] = self.session
        js['metadata'].update(dict([
            (
                name, 
                {
                    'type': 'String',
                    'default': '',
                    'desc': '',
                    'hidden': 'visible',
                    'value': value,
                    'required': False
                }
            ) for name, value in self.task_metadata.items()
        ]))
        

        # Supply sequence generators which are declared to be compatible with the selected task class
        exp_generators = dict()
        if hasattr(Exp, 'sequence_generators'):
            for seqgen_name in Exp.sequence_generators:
                try:
                    g = Generator.objects.using(self._state.db).get(name=seqgen_name)
                    exp_generators[g.id] = seqgen_name
                except:
                    print("missing generator %s" % seqgen_name)
        js['generators'] = exp_generators

        ## Add the sequence, used when the block gets copied
        print("getting the sequence, if any")
        if issubclass(self.task.get(), experiment.Sequence):
            js['sequence'] = {self.sequence.id:self.sequence.to_json()}

        datafiles = DataFile.objects.using(self._state.db).filter(entry=self.id)

        ## Add data files linked to this task entry to the web interface.
        backup_root = KeyValueStore.get('backup_root', '/None')

        js['datafiles'] = dict()
        system_names = set(d.system.name for d in datafiles)
        for name in system_names:
            js['datafiles'][name] = [d.get_path() + ' (backup status: %s)' % d.backup_status for d in datafiles if d.system.name == name]

        # Parse the "report" data and put it into the JS response
        js['report'] = self.offline_report()

        recording_sys_make = KeyValueStore.get('recording_sys')

        _neuralinfo = dict(is_seed=Exp.is_bmi_seed, length=0, name='', units=[])
        js['bmi'] = dict(_neuralinfo=_neuralinfo)
        if recording_sys_make == 'plexon':
            try:
                from plexon import plexfile # keep this import here so that only plexon rigs need the plexfile module installed
                plexon = System.objects.using(self._state.db).get(name='plexon')
                df = DataFile.objects.using(self._state.db).get(entry=self.id, system=plexon)

                _neuralinfo = dict(is_seed=Exp.is_bmi_seed)
                if Exp.is_bmi_seed:
                    plx = plexfile.openFile(df.get_path().encode('utf-8'), load=False)
                    path, name = os.path.split(df.get_path())
                    name, ext = os.path.splitext(name)

                    _neuralinfo['length'] = plx.length
                    _neuralinfo['units'] = plx.units
                    _neuralinfo['name'] = name

                js['bmi'] = dict(_neuralinfo=_neuralinfo)
            except MemoryError:
                print("Memory error opening plexon file!")
                js['bmi'] = dict(_neuralinfo=None)
            except (ObjectDoesNotExist, AssertionError, IOError):
                print("No plexon file found")
                js['bmi'] = dict(_neuralinfo=None)

        elif recording_sys_make == 'blackrock':
            try:
                print('skipping .nev conversion')
                js['bmi'] = dict(_neuralinfo=None)

                # length, units = parse_blackrock_file(self.nev_file, self.nsx_files, self)

                # js['bmi'] = dict(_neuralinfo=dict(
                #     length=length,
                #     units=units,
                #     name=name,
                #     is_seed=int(Exp.is_bmi_seed),
                #     ))

            except (ObjectDoesNotExist, AssertionError, IOError):
                print("No blackrock files found")
                js['bmi'] = dict(_neuralinfo=None)
            except:
                import traceback
                traceback.print_exc()
                js['bmi'] = dict(_neuralinfo=None)
        elif recording_sys_make == 'TDT':
            print('This code does not yet know how to open TDT files!')
            js['bmi'] = dict(_neuralinfo=None)
            #raise NotImplementedError("This code does not yet know how to open TDT files!")
        elif recording_sys_make == 'ecube':
            try:
                sys = System.objects.using(self._state.db).get(name='ecube')
                df = DataFile.objects.using(self._state.db).get(entry=self.id, system=sys)
                filepath = df.get_path()

                from riglib.ecube import parse_file
                _neuralinfo = dict(is_seed=Exp.is_bmi_seed)
                if Exp.is_bmi_seed:
                    try:
                        info = parse_file(str(df.get_path()))
                        path, name = os.path.split(df.get_path())
                        name, ext = os.path.splitext(name)

                        _neuralinfo['length'] = info.length
                        _neuralinfo['units'] = info.units
                        _neuralinfo['name'] = name
                    except:
                        _neuralinfo['is_seed'] = False

                js['bmi'] = dict(_neuralinfo=_neuralinfo)
            except ModuleNotFoundError:
                print("ecube reader not installed")
                js['bmi'] = dict(_neuralinfo=None)
            except (ObjectDoesNotExist, IOError):
                print("No ecube file found")
                js['bmi'] = dict(_neuralinfo=None)
        else:
            print('Unrecognized recording_system!')

        for dec in Decoder.objects.using(self._state.db).filter(entry=self.id):
            if 'bmi' not in js:
                print("Warning: found a decoder but not a recording system. Your recording system may not be set up correctly")
                js['bmi'] = dict(_neuralinfo=None)
            js['bmi'][dec.name] = dec.to_json()

        # collections info
        js['collections'] = []
        for col in TaskEntryCollection.objects.all():
            if self in col.entries.all():
                js['collections'].append(col.name)

        js['plot_files'] = dict()  # deprecated
        js['flagged_for_backup'] = self.backup
        js['visible'] = self.visible
        js['template'] = self.template
        entry_name = self.entry_name if not self.entry_name is None else ""
        js['entry_name'] = entry_name
        print("TaskEntry.to_json finished!")
        return js

    @property
    def plx_file(self):
        '''
        Returns the name of the plx file associated with the session.
        '''
        plexon = System.objects.get(name='plexon')
        try:
            df = DataFile.objects.get(system=plexon, entry=self.id)
            return os.path.join(df.system.path, df.path)
        except:
            import traceback
            traceback.print_exc()
            return 'noplxfile'

    @property
    def nev_file(self):
        '''
        Return the name of the nev file associated with the session.
        '''
        try:
            df = DataFile.objects.get(system__name="blackrock", path__endswith=".nev", entry=self.id)
            return df.get_path()
        except:
            try:
                df = DataFile.objects.get(system__name="blackrock2", path__endswith=".nev", entry=self.id)
                return df.get_path()
            except:
                return None
            #import traceback
            #traceback.print_exc()
            #return 'no_nev_file'
            #return None

    @property
    def nsx_files(self):
        '''Return a list containing the names of the nsx files (there could be more
        than one) associated with the session.

        nsx files extensions are .ns1, .ns2, ..., .ns6
        '''
        try:
            dfs = []
            for k in range(1, 7):
                df_k = DataFile.objects.filter(system__name="blackrock", path__endswith=".ns%d" % k, entry=self.id)
                dfs += list(df_k)

            return [df.get_path() for df in dfs]
        except:
            try:
                dfs = []
                for k in range(1, 7):
                    df_k = DataFile.objects.filter(system__name="blackrock2", path__endswith=".ns%d" % k, entry=self.id)
                    dfs+= list(df_k)
                return [df.get_path() for df in dfs]
            except:
                return None

    @property
    def name(self):
        '''
        Return a string representing the 'name' of the block. Note that the block
        does not really have a unique name in the current implementation.
        Thus, the 'name' is a hack this needs to be hacked because the current way of determining a
        a filename depends on the number of things in the database, i.e. if
        after the fact a record is removed, the number might change. read from
        the file instead
        '''
        recording_sys_make = KeyValueStore.get('recording_sys')

        if recording_sys_make == 'plexon':
            try:
                return str(os.path.basename(self.plx_file).rstrip('.plx'))
            except:
                return 'noname'
        elif recording_sys_make == 'blackrock':
            try:
                return str(os.path.basename(self.nev_file).rstrip('.nev'))
            except:
                return 'noname'
        else:
            raise Exception('Unrecognized recording_system!')

    @classmethod
    def from_json(cls, js):
        pass

    def get_decoder(self, dbname=None):
        """
        Get the Decoder instance associated with this task entry
        """
        params = eval(self.params)
        decoder_id = params['bmi']
        return Decoder.objects.using(dbname).get(id=decoder_id)

    @property
    def desc(self):
        """Get a description of the TaskEntry using the experiment class and the
        record-specific parameters """
        from . import json_param
        params = json_param.Parameters(self.params)

        if self.report is not None and len(self.report) > 0:
            report_data = json.loads(self.report)
        else:
            report_data = None
        try:
            Exp = self.task.get_base_class()
            return Exp.get_desc(params.get_data(), report_data)
        except:
            return "Error generating description"

    def get_data_files(self, dbname=None):
        return list(DataFile.objects.using(dbname).filter(entry_id=self.id))

    def get_data_files_dict(self, data_dir="", dbname=None):
        file_list = self.get_data_files(dbname=dbname)
        files = {}
        for df in file_list:
            files[df.system.name] = os.path.join(data_dir, df.system.name, os.path.basename(df.path))
        return files

    def get_data_files_dict_absolute(self, dbname=None):
        file_list = self.get_data_files(dbname=dbname)
        files = {}
        for df in file_list:
            files[df.system.name] = os.path.join(df.system.path, os.path.basename(df.path))
        return files

    def make_hdf_self_contained(self, dbname=None):
        '''
        If the entry has an hdf file associated with it, dump the entry metadata into it
        '''
        try:
            df = DataFile.objects.using(dbname).get(entry__id=self.id, system__name="hdf")
            h5file = df.get_path()
        except:
            print("No HDF file to make self contained")
            return False

        import h5py
        hdf = h5py.File(h5file, mode='a')
        print("Adding database metadata to hdf file:")
        print(h5file)

        # Add any task metadata
        hdf['/'].attrs["task_name"] = self.task.name
        hdf['/'].attrs["features"] = [f.name for f in self.feats.all()]
        hdf['/'].attrs["rig_name"] = KeyValueStore.get('rig_name', 'unknown')
        hdf['/'].attrs["block_number"] = self.id
        hdf['/'].attrs["subject"] = self.subject.name
        hdf['/'].attrs["experimenter"] = self.experimenter.name
        hdf['/'].attrs["date"] = str(self.date)
        hdf['/'].attrs["project"] = self.project
        hdf['/'].attrs["session"] = self.session
        if self.sequence is not None:
            hdf['/'].attrs["sequence"] = self.sequence.name
            hdf['/'].attrs["sequence_params"] = self.sequence.params
            hdf['/'].attrs["generator"] = self.sequence.generator.name
            hdf['/'].attrs["generator_params"] = self.sequence.generator.params
        hdf['/'].attrs["report"] = self.report
        hdf['/'].attrs["notes"] = self.notes
        hdf['/'].attrs["sw_version"] = self.sw_version

        # Link any data files
        data_files = []
        for df in DataFile.objects.using(dbname).filter(entry__id=self.id):
            data_files.append(df.get_path())
        hdf['/'].attrs["data_files"] = data_files
        hdf.close()

        # TODO save decoder parameters to hdf file, if applicable

        return True

class Calibration(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.PROTECT)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    system = models.ForeignKey(System, on_delete=models.PROTECT)

    params = models.TextField()

    def __str__(self):
        return "{date}:{system} calibration for {subj}".format(date=self.date,
            subj=self.subject.name, system=self.system.name)

    def __repr__(self):
        return self.__str__()

    def get(self):
        from .json_param import Parameters
        return getattr(calibrations, self.name)(**Parameters(self.params).params)

class AutoAlignment(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.TextField()

    def __unicode__(self):
        return "{date}:{name}".format(date=self.date, name=self.name)

    def get(self):
        return calibrations.AutoAlign(self.name)


def decoder_unpickler(mod_name, kls_name):
    if kls_name == 'StateSpaceFourLinkTentacle2D':
        kls_name = 'StateSpaceNLinkPlanarChain'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'StateSpaceEndptVel':
        kls_name = 'LinearVelocityStateSpace'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'State':
        mod_name = 'riglib.bmi.state_space_models'
    mod = importlib.import_module(mod_name)
    return getattr(mod, kls_name)


class Decoder(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    entry = models.ForeignKey(TaskEntry, on_delete=models.PROTECT)
    path = models.TextField()

    def __str__(self):
        return "Decoder[{date}:{name}] trained from {entry}".format(date=self.date, name=self.name, entry=self.entry)

    def __repr__(self):
        return self.__str__()

    def get_data_path(self, db_name=None):
        # data_path = KeyValueStore.get('data_path', '', dbname=db_name)
        # if len(data_path) == 0:
        #     print("Database path not set up correctly!")
        # return data_path
        return System.objects.using(db_name).get(name='bmi').path

    @property
    def filename(self):
        data_path = self.get_data_path()
        return os.path.join(data_path, self.path)

    def load(self, db_name=None, **kwargs):
        data_path = self.get_data_path()
        decoder_fname = os.path.join(data_path, self.path)

        if os.path.exists(decoder_fname):
            try:
                return pickle.load(open(decoder_fname, 'rb'), encoding='latin1')
            # try:
            #     fh = open(decoder_fname, 'r')
            #     unpickler = pickle.Unpickler(fh, **kwargs)
            #     unpickler.find_global = decoder_unpickler
            #     dec = unpickler.load() # object will now contain the new class path reference
            #     fh.close()

            #     dec.name = self.name
            #     return dec
            except:
                import traceback
                traceback.print_exc()
                return None
        else: # file not present!
            print("Decoder file could not be found! %s" % decoder_fname)
            return None

    def get(self):
        return self.load()

    def to_json(self):
        dec = self.get()
        decoder_data = dict(name=self.name, path=self.path)
        if not (dec is None):
            decoder_data['cls'] = dec.__class__.__name__
            if hasattr(dec, 'units'):
                decoder_data['units'] = dec.units
            else:
                decoder_data['units'] = []

            if hasattr(dec, 'binlen'):
                decoder_data['binlen'] = dec.binlen
            else:
                decoder_data['binlen'] = 0

            if hasattr(dec, 'tslice'):
                decoder_data['tslice'] = dec.tslice
            else:
                decoder_data['tslice'] = []

        return decoder_data

def parse_blackrock_file_n2h5(nev_fname, nsx_files):
    '''
    # convert .nev file to hdf file using Blackrock's n2h5 utility (if it doesn't exist already)
    # this code goes through the spike_set for each channel in order to:
    #  1) determine the last timestamp in the file
    #  2) create a list of units that had spikes in this file
    '''
    nev_hdf_fname = nev_fname + '.hdf'

    if not os.path.isfile(nev_hdf_fname):
        subprocess.call(['n2h5', nev_fname, nev_hdf_fname])

    nev_hdf = tables.openFile(nev_hdf_fname, 'r')

    last_ts = 0
    units = []

    #for key in [key for key in nev_hdf.get('channel').keys() if 'channel' in key]:
    chans = nev_hdf.root.channel
    chan_names= chans._v_children
    for key in [key for key in list(chan_names.keys()) if 'channel' in key]:
        chan_tab = nev_hdf.root.channel._f_getChild(key)
        if 'spike_set' in chan_tab:
            spike_set = chan_tab.spike_set
            if spike_set is not None:
                tstamps = spike_set[:]['TimeStamp']
                if len(tstamps) > 0:
                    last_ts = max(last_ts, tstamps[-1])
                else:
                    print('skipping ', key, ': no spikes')

                channel = int(key[-5:])
                for unit_num in np.sort(np.unique(spike_set[:]['Unit'])):
                    units.append((channel, int(unit_num)))
        else:
            print('skipping ', key, ': no spikeset')

    fs = 30000.
    nev_length = last_ts / fs
    nsx_lengths = []

    if nsx_files is not None:
        nsx_fs = dict()
        nsx_fs['.ns1'] = 500
        nsx_fs['.ns2'] = 1000
        nsx_fs['.ns3'] = 2000
        nsx_fs['.ns4'] = 10000
        nsx_fs['.ns5'] = 30000
        nsx_fs['.ns6'] = 30000

        NSP_channels = np.arange(128) + 1

        nsx_lengths = []
        for nsx_fname in nsx_files:

            nsx_hdf_fname = nsx_fname + '.hdf'
            if not os.path.isfile(nsx_hdf_fname):
                # convert .nsx file to hdf file using Blackrock's n2h5 utility
                subprocess.call(['n2h5', nsx_fname, nsx_hdf_fname])

            nsx_hdf = h5py.File(nsx_hdf_fname, 'r')

            for chan in NSP_channels:
                chan_str = str(chan).zfill(5)
                path = 'channel/channel%s/continuous_set' % chan_str
                if nsx_hdf.get(path) is not None:
                    last_ts = len(nsx_hdf.get(path).value)
                    fs = nsx_fs[nsx_fname[-4:]]
                    nsx_lengths.append(last_ts / fs)

                    break

    length = max([nev_length] + nsx_lengths)
    return length, units,

def parse_blackrock_file(nev_fname, nsx_files, task_entry, nsx_chan = np.arange(96) + 1):
    ''' Method to parse blackrock files using new
    brpy from blackrock (with some modifications). Files are
    saved as a ____ file?

    # this code goes through the spike_set for each channel in order to:
    #  1) determine the last timestamp in the file
    #  2) create a list of units that had spikes in this file
    '''
    from riglib.blackrock.brpylib import NevFile, NsxFile

    # First parse the NEV file:
    if nev_fname is not None:
        nev_hdf_fname = nev_fname + '.hdf'
        if task_entry is not None:
            datafiles = DataFile.objects.using(task_entry._state.db).filter(entry_id=task_entry.id)
            files = [d.get_path() for d in datafiles]
        else:
            # Guess where the file shoudl be:
            files = ['/storage/rawdata/blackrock/'+nev_hdf_fname]

        if nev_hdf_fname in files:
            hdf = tables.openFile(nev_hdf_fname)
            n_units = hdf.root.attr[0]['n_units']
            last_ts = hdf.root.attr[0]['last_ts']
            units = hdf.root.attr[0]['units'][:n_units]

        else:
            try:
                nev_file = NevFile(nev_fname)
                spk_data = nev_file.getdata()
            except:
                print('nev file is not available for opening. Try in a few seconds!')
                raise Exception

            # Make HDF file from NEV file #
            last_ts, units, h5file = make_hdf_spks(spk_data, nev_hdf_fname)

            if task_entry is not None:
                from . import dbq
                dbq.save_data(nev_hdf_fname, 'blackrock', task_entry.pk, move=False, local=True, custom_suffix='', dbname=task_entry._state.db)

        fs = 30000.
        nev_length = last_ts / fs
    else:
        units = []
        nev_length = 0

    tmax_cts = 0
    if nsx_files is not None:
        for nsx_fname in nsx_files:
            nsx_hdf_fname = nsx_fname + '.hdf'
            if not os.path.isfile(nsx_hdf_fname):

                # convert .nsx file to hdf file using Blackrock's n2h5 utility
                # subprocess.call(['n2h5', nsx_fname, nsx_hdf_fname])
                nsx_file = NsxFile(nsx_fname)

                # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
                cont_data = nsx_file.getdata(nsx_chan, 0, 'all', 1)

                # Close the nsx file now that all data is out
                nsx_file.close()

                # Make HDF file:
                tmax_cts = make_hdf_cts(cont_data, nsx_hdf_fname, nsx_file)
                if task_entry is not None:
                    from . import dbq
                    dbq.save_data(nsx_hdf_fname, 'blackrock', task_entry.pk, move=False, local=True, custom_suffix='', dbname=task_entry._state.db)
            else:
                tmax_cts = 0

    length = max([nev_length] + [tmax_cts])
    return length, units

def make_hdf_spks(data, nev_hdf_fname):
    last_ts = 0
    units = []

    #### Open h5file: ####
    tf = tempfile.NamedTemporaryFile(delete=False)
    h5file = tables.openFile(tf.name, mode="w", title='BlackRock Nev Data')
    h5file.createGroup('/', 'channel')

    ### Spike Data First ###
    channels = data['spike_events']['ChannelID']
    base_str = 'channel00000'
    for ic, c in enumerate(channels):
        c_str = base_str[:-1*len(str(c))]+ str(c)
        h5file.createGroup('/channel', c_str)
        tab = h5file.createTable('/channel/'+c_str, 'spike_set', spike_set)

        for i, (ts, u, wv) in enumerate(zip(data['spike_events']['TimeStamps'][ic], data['spike_events']['Classification'][ic], data['spike_events']['Waveforms'][ic])):
            trial = tab.row
            last_ts = np.max([last_ts, ts])
            skip = False
            if u == 'none':
                u = 10
            elif u == 'noise':
                skip = True

            if not skip:
                trial['Unit'] = u
                trial['Wave'] = wv
                trial['TimeStamp'] = ts
                trial.append()

        #Check for non-zero units:
        if len(data['spike_events']['TimeStamps'])>0:
            un = np.unique(data['spike_events']['Classification'][ic])
            for ci in un:
                #ci = 10
                if ci == 'none':
                    # Unsorted
                    units.append((c, 10))
                elif ci == 'noise':
                    pass
                else:
                    # Sorted (units are numbered )
                    units.append((c, int(ci)))

        tab.flush()

    ### Digital Data ###
    try:
        ts = data['dig_events']['TimeStamps']
        val = data['dig_events']['Data']

        for dchan in range(1, 1+len(ts)):
            h5file.createGroup('/channel', 'digital000'+str(dchan))
            dtab = h5file.createTable('/channel/digital000'+str(dchan), 'digital_set', digital_set)

            ts_chan = ts[dchan-1]
            val_chan = val[dchan-1]

            for ii, (tsi, vli) in enumerate(zip(ts_chan, val_chan)):
                trial = dtab.row
                trial['TimeStamp'] = tsi
                trial['Value'] = vli
                trial.append()
            last_ts = np.max([last_ts, tsi])
            dtab.flush()
    except:
        print('no digital info in nev file ')

    # Adding length / unit info:
    tb = h5file.createTable('/', 'attr', mini_attr)
    rw = tb.row
    rw['last_ts'] = last_ts
    U = np.zeros((500, 2))
    n_units = len(units)
    U[:n_units, :] = np.vstack((units))

    rw['units'] = U
    rw['n_units'] = n_units
    rw.append()
    tb.flush()

    h5file.close()
    shutil.copyfile(tf.name, nev_hdf_fname)
    os.remove(tf.name)
    print('successfully made HDF file from NEV file: %s' %nev_hdf_fname)

    un_array = np.vstack((units))
    idx = np.lexsort((un_array[:, 1], un_array[:, 0]))
    units2 = [units[i] for i in idx]

    return last_ts, units2, h5file

def make_hdf_cts(data, nsx_hdf_fname, nsx_file):
    last_ts = []
    channels = []

    tf = tempfile.NamedTemporaryFile(delete=False)
    h5file = tables.openFile(tf.name, mode="w", title='BlackRock Nsx Data')
    h5file.createGroup('/', 'channel')

    channel_ids = data['elec_ids']
    hdr_ids = data['ExtendedHeaderIndices']
    channel_labels = [nsx_file.extended_headers[h]['ElectrodeLabel'] for h in hdr_ids]
    print('saving channel nums: ', channel_ids)

    for ic, (c, chan) in enumerate(zip(channel_ids, channel_labels)):
        channel_data = data['data'][ic]
        c_str = 'channel'+str(c).zfill(5)
        h5file.createGroup('/channel', c_str)
        h5file.createArray('/channel/'+c_str, 'Value', channel_data)

    t = data['start_time_s'] + np.arange(data['data'].shape[1]) / data['samp_per_s']
    h5file.createArray('/channel', 'TimeStamp', t)
    #h5file.createArray('/channel', 'ElectrodeLabels', np.hstack((channel_labels)))
    h5file.close()
    shutil.copyfile(tf.name, nsx_hdf_fname)
    os.remove(tf.name)
    print('successfully made HDF file from NSX file: %s' %nsx_hdf_fname)
    return t[-1]

class spike_set(tables.IsDescription):
    TimeStamp = tables.Int32Col()
    Unit = tables.Int8Col()
    Wave = tables.Int32Col(shape=(48,))

class digital_set(tables.IsDescription):
    TimeStamp = tables.Int32Col()
    Value = tables.Int16Col()

class continuous_set(tables.IsDescription):
    TimeStamp = tables.Int32Col()
    Value = tables.Int16Col()

class mini_attr(tables.IsDescription):
    last_ts = tables.Float64Col()
    units = tables.Int16Col(shape=(500, 2))
    n_units = tables.Int16Col()


class DataFile(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    local = models.BooleanField(default=True)
    archived = models.BooleanField(default=False)
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System, on_delete=models.PROTECT, blank=True, null=True)
    entry = models.ForeignKey(TaskEntry, on_delete=models.PROTECT, blank=True, null=True)
    backup_status = models.CharField(max_length=256, blank=True, null=True)

    @staticmethod
    def create(system, task_entry, path, **kwargs):
        df = DataFile(system_id=system.id, entry_id=task_entry.id)

        for attr in kwargs:
            setattr(df, attr, kwargs[attr])

        post_processor = system.get_post_processor()
        df.path = post_processor(path)
        df.save()
        return df

    def __str__(self):
        if self.entry_id > 0:
            return "{name} datafile for {entry}".format(name=self.system.name, entry=self.entry)
        else:
            return "datafile '{name}' for System {sys_name}".format(name=self.path, sys_name=self.system.name)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return dict(system=self.system.name, path=self.path)

    def get(self):
        '''
        Open the datafile, if it's of a known type
        '''
        if self.system.name == 'hdf':
            import tables
            return tables.open_file(self.get_path())
        elif self.path[-4:] == '.pkl': # pickle file
            import pickle
            return pickle.load(open(self.get_path()))
        else:
            raise ValueError("models.DataFile does not know how to open this type of file: %s" % self.path)

    def get_path(self, check_archive=False):
        '''
        Get the full path to the file
        '''
        if not check_archive and not self.archived:
            return os.path.join(self.system.path, self.path)

        paths = self.system.archive.split()
        for path in paths:
            fname = os.path.join(path, self.path)
            if os.path.isfile(fname):
                return fname

        raise IOError('File has been lost! '+fname)

    def has_cache(self):
        if self.system.name != "plexon":
            return False

        path, fname = os.path.split(self.get_path())
        fname, ext = os.path.splitext(fname)
        cache = os.path.join(path, '.%s.cache'%fname)
        return os.path.exists(cache)

    def remove(self, **kwargs):
        try:
            os.unlink(self.get_path())
        except OSError:
            print("already deleted...")

    def delete(self, **kwargs):
        self.remove()
        super(DataFile, self).delete(**kwargs)

    def is_backed_up(self, backup_root):
        '''
        Return a boolean indicating whether a copy of the file is available on the backup
        '''
        try:
            fname = self.get_path()
            rel_datafile = os.path.relpath(fname, '/storage')
            backup_fname = os.path.join(backup_root, rel_datafile)
            return os.path.exists(backup_fname)
        except:
            return False

    @property
    def file_size(self):
        try:
            path = self.get_path()
            return os.stat(path).st_size
        except:
            print("Error getting data file size: ", self)
            traceback.print_exc()
            return -1

    def upload_to_cloud(self):
        """Upload file to google cloud storage"""
        full_filename = self.get_path()
        data = dict(full_filename=full_filename, filename=os.path.basename(full_filename),
            block_number=self.entry.id, msg_type='upload_file')
        cloud.send_message_and_wait(data)

    def verify_cloud_backup(self):
        """Check that cloud storage has this file with matching MD5 sum"""
        full_filename = self.get_path()

        import hashlib
        def md5(fname):
            hash_md5 = hashlib.md5()
            with open(fname, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        data = dict(full_filename=full_filename, filename=os.path.basename(full_filename),
            md5_hash=md5(full_filename), block_number=self.entry.id, msg_type='check_status')
        cloud.send_message_and_wait(data)


class TaskEntryCollection(models.Model):
    """ Collection of TaskEntry records grouped together, e.g. for analysis """
    name = models.CharField(max_length=200, default='')
    entries = models.ManyToManyField(TaskEntry)

    @property
    def safe_name(self):
        return self.name.replace(' ', '_')

    def add_entry(self, te):
        if not isinstance(te, TaskEntry):
            te = TaskEntry(id=te)

        if te not in self.entries.all():
            self.entries.add(te)

    def remove_entry(self, te):
        if not isinstance(te, TaskEntry):
            te = TaskEntry(id=te)

        if te in self.entries.all():
            self.entries.remove(te)


class KeyValueStore(models.Model):
    key = models.TextField()
    value = models.TextField()

    def __str__(self):
        return "KV[%s => %s]" % (self.key, self.value)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get(cls, key, default=None, dbname=None):
        try:
            if dbname is not None:
                objs = cls.objects.using(dbname).filter(key=key)
            else:
                objs = cls.objects.filter(key=key)
            if len(objs) == 0:
                return default
            if len(objs) == 1:
                return objs[0].value
            if len(objs) > 1:
                raise ValueError("Duplicate keys: %s" % key)
        except:
            print("KeyValueStore error")
            print("key = %s, default=%s, dbname=%s" % (key, default, dbname))
            traceback.print_exc()
            return default

    @classmethod
    def set(cls, key, value, dbname=None):
        if dbname is not None:
            matching_recs = cls.objects.using(dbname).filter(key=key)
        else:
            matching_recs = cls.objects.filter(key=key)
        if len(matching_recs) == 0:
            obj = cls(key=key, value=value)
            obj.save()
        elif len(matching_recs) == 1:
            obj = matching_recs[0]
            obj.value = value
            obj.save()
        else:
            raise ValueError("Duplicate keys: %s" % key)

