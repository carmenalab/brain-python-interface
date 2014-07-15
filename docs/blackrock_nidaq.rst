.. _blackrock_nidaq:

Blackrock NSP digital input
===========================

Right now, this is a just a collection of notes about the 

We use the Blackrock NSP's digital input port for 2 purposes:
1) Remote recording control (starting/stopping/resuming recording remotely)
2) Writing information into the digital input port during the experiment so that timestamps corresponding to the arrival of various data (ArmAssist feedback data, ReHand feedback data, EMG data) are saved into the Blackrock .nev file for later
comparison with spike timestamps.

The digital input port has 16 data pins (D0 through D15) and one digital strobe (DS) pin.


See the NeuroPort manual for more details.