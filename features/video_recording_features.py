'''
For capturing frames from a camera
'''
import time
import os
import traceback
from riglib.experiment import traits
from riglib.mp_calc import MultiprocShellCommand
from datetime import datetime
from riglib.e3vision import E3VisionInterface

class E3Video(traits.HasTraits):
    '''
    Enables recording of e3vision cameras from White-Matter.
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        '''
        super().init()

        # Start the natnet client and recording
        now = datetime.now()
        take = now.strftime("Take %Y-%m-%d %H_%M_%S")
        self.e3v_status = 'offline'
        try:
            if self.saveid is not None:
                take += " (%d)" % self.saveid
                self.e3v_filename = take
                e3v = E3VisionInterface(take)
                e3v.update_camera_status()
                e3v.start_rec()
                self.e3v_status = 'recording'
            else:
                e3v = E3VisionInterface()
                e3v.update_camera_status()
                self.e3v_status = 'online'
            self.e3v = e3v
        except Exception as e:
            self.e3v_status = traceback.format_exc()

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run().
        '''
        if not self.e3v_status in ['recording', 'online']:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.e3v_status)
            self.termination_err.seek(0)
            self.state = None
            super().run()
            try:
                self.e3v.stop_rec()
            except:
                pass
        else:
            try:
                super().run()
            finally:
                if self.e3v_status == 'recording':
                    print("Stopping e3v recording")
                    self.e3v.stop_rec()

    def cleanup(self, database, saveid, **kwargs):
        '''
        Save the e3v recorded filenames into the database. Only gets called when saveid is not None.
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        print("Saving WM e3vision files to database...")
        try:
            suffix = '' # note: database functions don't take keyword arguements like custom_suffix=suffix
            database.save_data(self.e3v_filename, "e3v", saveid, False, False, suffix) # Make sure you actually have an "e3v" system added!
        except Exception as e:
            print(e)
            return False
        print("...done.")
        return super_result
    

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
