import os
import datetime
import numpy as np
from open_ephys.control import OpenEphysHTTPServer

IP_neuropixel = '10.155.205.108'
gui = OpenEphysHTTPServer(IP_neuropixel)

print(datetime.date.today())
prepend_text = datetime.date.today()
filename = '_Neuropixel_'
append_text = 'test6'
gui.set_prepend_text(prepend_text)
gui.set_base_text(filename)
gui.set_append_text(append_text)
#gui.set_start_new_dir() # This isn't necessary because saved directory changes when file name changes
#gui.record(10)

print(gui.get_recording_info())
print(gui.get_processors('Neuropix-PXI'))
A = gui.get_parameters(105,0)
print(A)
#print(gui.get_parameters(103,1))