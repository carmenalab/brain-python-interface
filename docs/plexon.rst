======
Plexon
======
Plexon has graciously provided some C headers for their stuff. Unfortunately, most of it is poorly organized, and ripped verbatim from incomplete code, so much of it had to be reverse engineered.


Plexfile
========
Included with the plexon library is a Plexfile module, which allows the quick reslicing of \*.plx files.

To open a plexon file::

    from riglib.plexon import plexfile
    plx = plexfile.openFile("demo.plx")

To read the spikes from time 0:10 seconds, ::

    plx.spikes[:10]
    
Unlike normal python slicing, decimal slices are ok::

    plx.spikes[0.5:2.5]
    
**Be aware of how much time you slice, as you will QUICKLY run out of memory!**

For continuous data, use the corresponding names *wideband*, *spkc*, *lfp*, *analog*::

    plx.lfp[0.5:2.5]
    
This will slice from 0.5 to 2.5 s for all available LFP channels. If you'd like to slice from individual channels, use typical python slice semantics::

    touch = plx.analog[0.5:2.5, 0]          # retrieve channel AN01
    data1 = plx.wideband[0.5:2.5, 140:160]  # retrieve channels WB141 - WB160
    data2 = plx.spkc[0.5:2.5, [145, 161]]   # retrieve channels SPKC146 and SPKC162

