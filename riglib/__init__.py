'''
Declarations for importing riglib as a module. Currently unused
'''

class FuncProxy(object):
	'''
	This class is similar to tasktrack.FuncProxy but appears to be unused
	'''
    def __init__(self, name, pipe, event):
        self.pipe = pipe
        self.name = name
        self.event = event

    def __call__(self, *args, **kwargs):
        self.pipe.send((self.name, args, kwargs))
        self.event.set()
        return self.pipe.recv()
