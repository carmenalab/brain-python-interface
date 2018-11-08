'''

'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os
import subprocess
import multiprocessing as mp 
from riglib.experiment import traits

###### CONSTANTS
sec_per_min = 60




class MultiprocShellCommand(mp.Process):
    '''
    Execute a blocking shell command in a separate process
    '''
    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd
        self.done = mp.Event()
        super(MultiprocShellCommand, self).__init__(*args, **kwargs)

    def run(self):
        '''
        Docstring
        '''
        import os
        os.popen(self.cmd)
        self.done.set()

    def is_done(self):
        return self.done.is_set()


class SingleChannelVideo(traits.HasTraits):
    def __init__(self, *args, **kwargs):
        super(SingleChannelVideo, self).__init__(*args, **kwargs)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' spawns process to run ssh command to begin video recording
        '''
        self.video_basename = 'video_%s.avi' % time.strftime('%Y_%m_%d_%H_%M_%S')        
        self.device_name = '/dev/video0'
        cmd = "ssh -tt video /home/lab/bin/recording_start.sh %s %s" % (self.device_name, self.video_basename)
        
        cmd_caller = MultiprocShellCommand(cmd)
        cmd_caller.start()
        self.cmd_caller = cmd_caller
        print("started video recording")
        print(cmd)
        
        super(SingleChannelVideo, self).init()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Stop video recording and link file to database

        Parameters
        ----------
        database
        saveid

        Returns
        -------
        None
        '''

        print("executing command to stop video recording")
        os.popen('ssh -tt video /home/lab/bin/recording_end.sh')

        print("Checking if video recording is done")
        if self.cmd_caller.done.is_set():
            print("SSH command finished!")

        super(SingleChannelVideo, self).cleanup(database, saveid, **kwargs)
        print("sleeping so that video recording can wrap up")
        time.sleep(5)        

        ## Get the video filename
        video_fname = os.path.join('/storage/video/%s' % self.video_basename)

        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(video_fname, "video", saveid)
        else:
            database.save_data(video_fname, "video", saveid, dbname=dbname)
