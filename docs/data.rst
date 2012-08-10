..	_data:

Reading saved data
==================

Plexon files
------------

Neural data, any attached analog input data, and synchronization markers for accompanying HDF5 files are stored in Plexon's .plx file format.

Plexon has graciously provided some C headers for their stuff. Unfortunately, most of it is poorly organized, and ripped verbatim from incomplete code, so much of it had to be reverse engineered.

Using Plexfile
>>>>>>>>>>>>>>
Included with the plexon library is a Plexfile module, which allows the quick reslicing of \*.plx files.

To open a plexon file::

    from riglib.plexon import plexfile
    plx = plexfile.openFile("demo.plx")

To read all the spikes from time 0:10 seconds, ::

    plx.spikes[:10]
    
Unlike normal python slicing, decimal slices are ok::

    plx.spikes[0.5:2.5]
    
**Be aware of how much time you slice, as you will QUICKLY run out of memory!**

A slice like the above will return a record array with three data types: *ts*, *chan*, and *unit*.

To retrieve the timestamps where spikes occurred within a time slice, along with the corresponding channel and unit numbers::

    spike_times = plx.spikes[0.5:2.5]['ts']
    spike_chans = plx.spikes[0.5:2.5]['chan']
    spike_units = plx.spikes[0.5:2.5]['unit']

You can filter out data for a particular unit using logical combinations of the unit and channel labels. Channels here are labeled 1-256 to match the Plexon convention, but unit labels have been changed from letters to numbers (0 = unsorted, 1 = a, 2 = b, etc.). For example::

    spikes = plx.spikes[0.5:2.5]
    unit161a = spikes[np.logical_and(spikes['chan'] == 161, spikes['unit'] == 1)]['ts']

will return the timestamps of all of unit 161a's spikes between .5 and 2.5s.

For continuous data, use the corresponding names *wideband*, *spkc*, *lfp*, *analog*. Timestamps and data values can be extracted independently from a slice, and both are returned as a *time* x *channel* array. For example::

    plx.lfp[.5:2.5].time
    plx.lfp[.5:2.5].data

will return the timestamps and data respectively from 0.5 to 2.5 s for all available LFP channels in the file. If you'd like to slice from individual channels, use typical python slice semantics::

    touch = plx.analog[0.5:2.5, 0].data          # retrieve channel AN01 data
    data1 = plx.wideband[0.5:2.5, 140:160].data  # retrieve channels WB141 - WB160 data
    data2 = plx.spkc[0.5:2.5, [145, 161]].time   # retrieve channels SPKC146 and SPKC162 timestamps

HDF5 files
----------

Data from all attached systems except Plexon, as well as the state transition logs are stored in .hdf5 files.