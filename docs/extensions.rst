.. _extensions:

Experiment extensions
=====================

These are built in classes with added special structure or functionality on top of the basic :class:`Experiment` class. They can be used as the basis for specific tasks that require this functionality. Task classes can inherit from more than one of these (e.g. a task can be both a sequence and logged).

Logged experiment (:class:`logexperiment`)
------------------------------------------

Any experiment that requires a log of state transitions to be saved. Most experiments fall into this category.

Sequence experiment (:class:`sequence`)
---------------------------------------

An experiment that has a trial state that is dependent on a parameter that changes for each new trial. Requires a generator.

Generators
>>>>>>>>>>

The *wait* state
>>>>>>>>>>>>>>>>

Trial types experiment (:class:`trialtypes`)
--------------------------------------------