'''
Code for interacting with the Phdigets API
'''
import time
import itertools
import numpy as np
from .source import DataSourceSystem



    # shoudl be 
## Installation 
## https://github.com/signal11/hidapi
## git clone
## comment out macports (m4) installation from .bash_profile: 

# "# MacPorts Installer addition on 2013-11-19_at_19:41:04: adding an appropriate PATH variable for use with MacPorts.
# export PATH=/opt/local/bin:/opt/local/sbin:$PATH <--- comment this line 

# Then cd ~/hidapi
# ./bootstrap
# ./configure
# make
# make install 

# Then clone python bindings; https://github.com/awelkie/pyhidapi
# got 'python' pointer from .bash_aliases: 
# sudo /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python setup.py install

# Had to go into the python file in /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
# where hidapi is installed, and point it to the correct library (/usr/local/lib)
# setting the DLDY_LIBRARY thing caused havoc in other placse

class TabletSystem(DataSourceSystem):
    '''
    Generic DataSourceSystem interface for the tablets
    '''
    update_freq = 200
    dtype = np.dtype((np.float, (2,)))

    def __init__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.interval = 1. / self.update_freq
        self.data = np.zeros((2,))
        
        import hidapi
        hidapi.hid_init()
        self.tablet = hidapi.hid_open(0x28bd,0x0913)
        tmp = hidapi.hid_read(self.tablet, 8)
        self.tic = time.time()

        # tablet = hidapi.hid_open(0x28bd,0x0913)
        # tmp = hidapi.hid_read(tablet, 8)

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass
    
    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        import hidapi
        toc = time.time() - self.tic
        if 0 < toc < self.interval:
            time.sleep(self.interval - toc)
        else:
            byte_array = hidapi.hid_read_timeout(self.tablet, 8, 5)
            if len(byte_array) > 0:
                x = [int(ba) for ba in byte_array]
                xx = x[3]*255 + x[2]
                yy = x[5]*255 + x[4]

                ### convert to CM
                pix = np.array([xx, 7591 - yy])
                inches = pix/2530.25 ### Rough converion factor 
                self.tic = time.time()
                return inches
    
    def sendMsg(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.tablet.close()

