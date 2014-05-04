.. _bmi

Brain-machine interface (BMI) code
==================================

The BMI code is split between the "tasks" module and the "bmi" module. The "bmi" module contains
the "low-level" components of the BMI, including the decoding algorithm, the assist methods, 
the adaptive filtering techinques for parameter fitting, etc. The "tasks" module integrates these
components to enable the BMI subject to perform a target-capture task.

..  automodule:: riglib.bmi.bmi
    :members:

System architecture
===================

Simulating BMI
==============
BMI simulation involve the the "task" program as well as the BMI software. After installing the
software, control of a 2D cursor can be simulated by running the script

run $HOME/code/bmi3d/tests/sim_clda/sim_clda_multi.py --alg=RML

where RML is an example of a CLDA algorithm that can be simulated using the script. The basic premise
behind all of the implemented simulations (as of May 2014) is that the spike rates/time-stamps of 
a population of neurons as a response to the stimulus of "intended" BMI state change, or intended
kinematics. Intention is simulated as a feedback controller. (see riglib.bmi.feedback_controllers
for examples). 

Training a Decoder
==================
There are at least two contexts in which one would need to "train" (as opposed to "re-train" or "adapt") a 
Decoder. The first is to create an entirely new set of Decoder parameters from a "seeding" session.
For instance, it is common to create a new decoder based on the neural response to subjects
watching cursor movements without any control over the cursor (i.e. a "visual feedback" task). 
A second case where one would want to create a new Decoder object might be to do a "batch" recalibration
(Gilja et al., 2012). 

Functions to train new decoder objects are in the module riglib.bmi.train:

..  automodule:: riglib.bmi.train
    :members: