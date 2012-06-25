..	_tasks:

Creating tasks
==============

The definition of a task class contains a set of states and the rules for moving between those states, a set of parameters (both hard-coded and user-defined), state methods (which specify what happens during each state), and transition methods (which specify the conditions for changing states). Every task is a subclass of the :class:`Experiment` class.

The state transition diagram
----------------------------

A state can be thought of as a discrete part of the task which is triggered by some condition being met and ends when some other condition is met (i.e. waiting for fixation, or a target hold). The state transition diagram defines the structure of the task. For each possible state it lists all the possible subsequent states and the events that trigger those transitions.

For example, the parent class :class:`Experiment` has the following structure, where ovals represent states and arrows represent transitions:

..	image:: states.png

A state transition diagram is written in the code as a nested dictionary with the name ``status`` that pairs each state with a dictionary containing all possible event-next state transitions that could occur from that state. For the task illustrated above, the state transition structure in the code looks like::

	status = dict(
	        wait = dict(start_trial="trial", premature="penalty", stop=None),
	        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
	        reward = dict(post_reward="wait"),
	        penalty = dict(post_penalty="wait"),
	    )

The state transition diagram is usually the very first thing defined in a task class.

Parameters
----------

Parameters that are set by the user when the task is run can be defined as `Traits <http://code.enthought.com/projects/traits/>`_ within the class definition for the task::

    #settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")

The first argument is the default value for the trait and the second is a docstring that will show up when the mouse hovers over that parameter in the web interface. The name of the variable will be the text the user sees in the list of parameters (in this case ``reward_time``.

Methods
-------

A task class' methods determine the behavior within each state and the criteria for triggering state transitions. There are five types of special methods:

__init__
>>>>>>>>

Every task has an __init__ method that contains actions to be performed once when the task is first run::

    #initialize and create fixation point object
    def __init__(self, **kwargs):
        super(FixationTraining, self).__init__(**kwargs)
        self.fixation_point = Sphere(radius=.1, color=(1,0,0,1))

If no initialization steps are necessary for the task, the __init__ method can be omitted (because it will automatically inherit the parent __init__ method). This is true of the other special methods as well; however, if an __init__ method is included, it MUST contain a call to the parent method, whereas the rest of the special methods may be written to replace the parent methods if desired.

_start_
>>>>>>>

_start_ methods specify actions to be performed at the onset of a new state::

def _start_wait(self):
        super(TargetCapture, self)._start_wait()
        #set target color
        self.origin_target.color = (1,0,0,.5)
        #hide target from previous trial
        self.origin_target.detach()
        self.requeue()

In the above example, every time the task enters the *wait* state, the origin target's color changes and the target is hidden from the screen, in addition to whatever actions are already performed by the parent class' _start_wait method.

The full name of the method should be the ``_start_`` prefix followed by a state name that appears in the state transition diagram. (This goes for _while_ and _end_ methods as well.)

_while_
>>>>>>>

_while_ methods specify actions to be repeated (usually once per frame) while the task is in a state::

    def _while_wait(self):
        self.update_cursor()

Here, the cursor position is being constantly updated during the *wait* state.

Experiment extensions
---------------------

make new page

Sequence
>>>>>>>>

Generators
<<<<<<<<<<

Special states
<<<<<<<<<<<<<<

TrialTypes
>>>>>>>>>>

Display technologies
--------------------

make new page

Pygame
>>>>>>

StereoOpenGL
>>>>>>>>>>>>