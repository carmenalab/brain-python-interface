import string
import inspect
import re

def parse_str(args_and_kwargs):
    args_and_kwargs = args_and_kwargs.split(',')
    kwargs = [x for x in args_and_kwargs if '=' in x]
    args = [x for x in args_and_kwargs if not '=' in x]
    kwargs = [kwarg.split('=') for kwarg in kwargs]
    return _parse(args, kwargs)

def _parse_type(s):
    s = str(s)
    m = re.match("<type '(\w+)'>", s)
    return m.group(1)


def _parse(args, kwargs, varargs=None, keywords=None, offset=''):
    ds = '''
    Docstring

    Parameters
    ----------
'''

    # Not sure why this hack is necessary...
    # ds += ' '
    for arg in args:
        if arg == 'self':
            continue
        ds += '''    %s : DATA_TYPE
        ARG_DESCR
''' % arg

    for arg, default in kwargs:
        ds += '''    %s : %s, optional, default=%s
        ARG_DESCR
''' % (arg, _parse_type(type(default)), default)


    if not (keywords is None):
        ds += '''    %s : optional kwargs
        ARG_DESCR
''' % (keywords)

    ds += '''
    Returns
    -------
'''

    ## Add a spacing offset to each line
    ds_ = ''
    for line in ds.split('\n'):
        ds_ += offset + line + '\n'

    return ds_


def parse_callable(obj, *args, **kw):
    argspec = inspect.getargspec(obj)
    if argspec.defaults is None:
        args = argspec.args
        kwargs = []
    else:
        args = argspec.args[:-len(argspec.defaults)]
        kwarg_names = argspec.args[-len(argspec.defaults):]
        kwarg_vals = argspec.defaults
        kwargs = list(zip(kwarg_names, kwarg_vals))

    return _parse(args, kwargs, argspec.varargs, argspec.keywords, **kw)
    
if __name__ == '__main__':
    parse_str('A, current_states, full_state_ls, axis=0')
