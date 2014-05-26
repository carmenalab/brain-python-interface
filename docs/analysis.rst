.. _analysis:

Data analysis code
------------------

Analysis of neural data often feels like busywork. This document describes code part of the BMI3D library intended to cut down on such busywork. 

The basic architecture
======================
Experiments run with the BMI3D experimental software are automatically logged into a database using the python package django. Central to the analysis of experimental data is the TaskEntry
model in the Django database (db.tracker.TaskEntry). However, it's not desireable to be 
manipulating the database code frequently since it's critical to the recording of data, so we mostly put our analysis code in a different place:

.. autoclass:: db.dbfunctions.TaskEntry
	:members: proc

basicanalysis
=============
..  automodule:: analysis.basicanalysis
    :members: