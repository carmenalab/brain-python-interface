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

Here's a simple python example that doesn't work::

    # Test case for CLDARecomputeParameters, to show non-blocking properties
    # of the recomputation
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    work_queue.put((None, None, None))

    clda_worker = CLDARecomputeParameters(work_queue, result_queue)
    clda_worker.start()

    while 1:
        try:
            result = result_queue.get_nowait()
            break
        except:
            print 'stuff'
        time.sleep(0.1)

Database integration with Django
--------------------------------
The ability of this library to do automatic data logging and file linking is a great feature of this library (and arguably, makes its complexity worthwhile). We recommend that you go through the introductory Django tutorial (https://docs.djangoproject.com/en/dev/intro/tutorial01/), which shows you how to build a simple Django application fairly similar to what we do during the experiment logging (though with a less complicated user interface).


Helpful python hints
====================
Some clues on the more tricky/magical aspects of the code

* ``getattr``
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
