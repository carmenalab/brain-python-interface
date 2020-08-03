class Singleton(object):
    '''Python implementation of the Singleton design pattern'''
    __instance = None

    @classmethod
    def get_instance(cls):
        instance_attr_name = '_%s__instance' % cls.__name__
        if getattr(cls, instance_attr_name) is None:
            cls()
        return getattr(cls, instance_attr_name)

    def __init__(self, *args, **kwargs):
        instance_attr_name = '_%s__instance' % self.__class__.__name__
        if getattr(self.__class__, instance_attr_name) is None:
            setattr(self.__class__, instance_attr_name, self)
        else:
            raise Exception("can't reinstantiate %s because it is a singleton!" % self.__class__.__name__)

