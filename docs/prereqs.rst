Assumed Python knowledge
========================
In order to use this library well, it is assumed that you are pretty familiar with Python. For example, some of the medium concepts you should at least recognize include

	- numpy/scipy
	- how to make a python module
	- "New" style inheritance and object-oriented programming in Python
	- iterators and generators

As with any software package in development, reading error messages and subsequent use of googling relevant portions of the message cannot be avoided.

A deep understanding of the core functionality of this library requires knowledge of a few concepts which are not typically found in brief introductions to Python and which are uncommon in scientific data analyses. Once your experiment is all set up and debugged, it's certainly possible for the experimenter to not need to know what's under the hood. However, depending on how complicated your experiment setup is, you may want to spend some time learning some of the other libraries/functionality that this library is built on top of.


Multiple inheritance
--------------------
Multiple inheritance is a contentious topic, with large disagreement between various internet talking heads about whether it is necessary or should even exist. Your opinion on the matter is irrelevant. This library uses multiple inheritance and a metaclass constructor in order to embed user-selected 'features' into 'experiments' at runtime. There certainly are other implementation strategies which do not emply multiple inheritance, but what's done is done. It would take much much less time to just learn multiple inheritance than it would to redo the library, and the effort would be pointless. Having an experimental setup where you can dynamically select or deselect large portions of functionality at runtime is going to be complicated no matter what you do. 

That's enough philosophy. As an obvious prerequisite to understanding multiple inheritance, you should understand single inheritance! This is a good introduction: http://learnpythonthehardway.org/book/ex44.html

This library uses new-style inheritance, where all classes inherit from the built-in type ``object``. A key reason for inheritance to exist is so that children can choose to either call, modfiy, or override their parent's methods. In classes inheriting from ``object``, the "parent's" attributes are accessed by use of the ``super`` function. 

``super`` lets you get your parent's methods without knowing the name of your parent, which may be nice if you ever change the inheritance tree. (In Python 2, you still have to refer to your own class name, so annoying things can happen if you change the name of the child class or copy-paste what should be straightforward code from another class). 

With multiple inheritance, there is not a single "parent" (hence the quotes). Python instead creates a "method resolution order" (MRO), a unique ordering of the parents methods to go through one at a time. When you call ``super``, you're really just asking for the next method in the MRO, rather than an explicit parent. The object will not be constructed if a consistent MRO cannot be defined. 

This is a fairly advanced description of what ``super`` can do: https://rhettinger.wordpress.com/2011/05/26/super-considered-super/. It's written for Python 3 rather than Python 2 (not backwards compatible) so it's more for reading than trying. 


Using multiple processes
------------------------
In experiments with many peripheral devices, data must be acquired and logged asynchronously. In this library, the 'multiprocessing' module is the main workhorse for enabling asynchronous input/output functionality. In particular, multiprocessing is used for the webserver to spawn tasks (``db.tasktrack``), for acquiring data from sources (``riglib.source``) and for any BMI computation which cannot complete in the relatively short event loop (``riglib.mp_calc``). 

This is a decent introduction to multiprocessing: https://pymotw.com/2/multiprocessing/basics.html

Here's a simple python example::

    import multiprocessing as mp
    import time

    #### Version 1: single-threaded
    def target_fn():
        time.sleep(2) # simulate thinkin
        print "TARGET FUNCTION: done computing answer"
        return "answer"

    t_start = time.time()
    print "Single-process version"
    target_fn()
    for k in range(30):
        print "fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start)
        time.sleep(0.1)

    print "\n\n\n\n\n\n"

    #### Version 2: multi-threaded
    t_start = time.time()
    print "Multi-process version"
    proc = mp.Process(target=target_fn)
    proc.start()
    for k in range(30):
        print "fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start)
        time.sleep(0.1)


    print "\n\n\n\n\n\n"

    #### Version 3: multi-threaded, alternate implementation
    class TargetClass(mp.Process):
        def run(self):
            target_fn()

    t_start = time.time()
    print "Multi-process version"
    p = TargetClass()
    p.start()
    for k in range(30):
        print "fake event loop, index %d, time since start of loop: %g" % (k, time.time() - t_start)
        time.sleep(0.1)

In the first single-threaded example, note that the whole loop is stalled until ``target_fn`` finishes running. In the multi-process version, the event loop runs at roughly the time one expects and ``target_fn`` emits its answer once it's done, without stalling any of the other things that might be happening in the event loop.

The third version is actually the same as the second. It just looks a bit more object-oriented, but as you can tell from the output, it accomplishes the same functionality. 

NOTE: similar functionality can be accompished with threads.

Database integration with Django
--------------------------------
The ability of this library to do automatic data logging and file linking is a great feature of this library (and arguably, makes its complexity worthwhile). We recommend that you go through the introductory Django tutorial (https://docs.djangoproject.com/en/dev/intro/tutorial01/), which shows you how to build a simple Django application fairly similar to what we do during the experiment logging (though with a less complicated user interface).


Helpful python hints
====================
Some clues on the more tricky/magical aspects of the code

* ``getattr`` and ``setattr``
    If you have an object ``obj`` with attribute ``param``, the two lines below are equivalent::
        
        In [3]: class TestClass(object):
           ...:     def __init__(self):
           ...:         self.param = 'value'
           ...:         

        In [4]: obj = TestClass()

        In [5]: obj.param
        Out[5]: 'value'

        In [6]: getattr(obj, 'param')
        Out[6]: 'value'

    The second one allows you to get an attribute of an object by specifying a string name. This can be useful in selecting which attribute of the object you want on the fly.  

* Making python modules

    Putting a ``__init__.py`` into a folder makes a module. Why is it sometimes necessary to make modules? Consider the following example::

        In [1]: pwd
        Out[1]: u'/Users/sgowda/code/bmi3d'

        In [2]: ls docs
        ...
        create_docstring.py             
        ...

        In [3]: from docs import create_docstring
        ---------------------------------------------------------------------------
        ImportError                               Traceback (most recent call last)
        <ipython-input-3-01a5608ea874> in <module>()
        ----> 1 from docs import create_docstring

        ImportError: No module named docs

    We tried to import the python file ``create_docstring.py``, which ``ipython`` can clearly see, but the import doesn't work! Putting an empty ``__init__.py`` in the docs directory tells python to treat the directory as a module and allows us to import a sub-module from inside the module. 

* Variable unpacking

    In the great "MATLAB vs Python" debate, one point clearly in the python column is its ability to deal with variable unpacking:

        In [1]: a, b, c = (1, 2, 3)

        In [2]: a
        Out[2]: 1

        In [3]: b
        Out[3]: 2

        In [4]: c
        Out[4]: 3

    This works in a variety of cases. For example, you can simultaneously iterate over two variables::

        In [5]: data = [(1, 10), (2, 20), (3, 30), (4, 40)]

        In [6]: for x, y in data:
           ...:     print x + y
           ...:     
        11
        22
        33
        44

    This also works for multiple output arguments from a function::

        In [7]: def fn():
           ...:     return 1, 3, 4
           ...: 

        In [8]: resp = fn()

        In [9]: resp
        Out[9]: (1, 3, 4)

        In [10]: a, b, c = fn()

        In [11]: a
        Out[11]: 1

        In [12]: b
        Out[12]: 3

        In [13]: c
        Out[13]: 4    

    One odd corner case to keep in mind is what happens when you return one argument but you try to return multiple arguments::

        In [14]: def fn2():
           ....:     return (1,)
           ....: 

        In [15]: resp = fn2()

        In [16]: resp
        Out[16]: (1,)

        In [17]: a, = fn2() # note the comma!

        In [18]: a
        Out[18]: 1

    Unlike in MATLAB, a length-1 object is NOT automatically demoted to a scalar in python


* @property decorator

* keyword arguments

    Keyword arguments in python are arguments which can be indexed by name. (Standard function call arguments are, by contrast, positional arguments). An example::

        In [2]: def fn(a=1, b=2):
           ...:         print a, b
           ...:     

        In [3]: fn()
        1 2

        In [4]: fn(a=3)
        3 2

        In [5]: fn(b=4)
        1 4

        In [6]: fn(b=3, a=6)
        6 3

    Keyword arguments let you specify function arguments in an order-independent manner, so you don't have to remember the exact order of arguments all the time. In addition, they let you supply default values for arguments so not every function call needs to explicitly list out all the arguments, to avoid repetition. Sometimes it's useful for a function to not specify an exhaustive list of keyword arguments it expects. For example::

        In [7]: def fn2(a=1, b=2, **kwargs):
           ...:     print a, b
           ...:     print kwargs
           ...:     

        In [8]: fn2(a=1, b=2, c=3)
        1 2
        {'c': 3}

    We gave an extra keyword argument ``c`` which was not used by the function. As shown by the output of the function call, this gets packed into a dictionary ``kwargs``. 

    This can be useful if you want to just pass all the extra keyword arguments to another function, without needing to explicitly name them in the top-level function. 