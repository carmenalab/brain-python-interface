import time
import os
import numpy as np
from datetime import datetime
from riglib.experiment import traits
from riglib import ecube, source
from features.neural_sys_features import CorticalBMI, CorticalData
import traceback

ecube_path = "/media/NeuroAcq" # TODO this should be configurable elsewhere
log_path = os.path.join(os.path.dirname(__file__), '../log')
log_filename = os.path.join(log_path, "ecube_log")

def make_ecube_session_name(saveid):
    now = datetime.now()
    session = now.strftime("%Y-%m-%d")
    number = str(saveid) if saveid else "Test"
    return session + "_BMI3D_te" + number

def log_str(s, mode="a", newline=True):
    if newline and not s.endswith("\n"):
        s += "\n"
    with open(log_filename, mode) as fp:
        fp.write(s)

def log_exception(err, mode="a"):
    with open(log_filename, mode) as fp:
        traceback.print_exc(file=fp)
        fp.write(str(err))

class RecordECube(traits.HasTraits):

    record_headstage = traits.Bool(False, desc="Should we record headstage data?")
    headstage_connector = traits.Int(7, desc="Which headstage input to record (1-indexed)")
    headstage_channels = traits.Tuple((1, 1), desc="Range of headstage channels to record (1-indexed)")
    ecube_status = None

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        
        # Check that we actually started recording
        if not self.ecube_status == "recording":
            return super_result
        
        ecube_session = make_ecube_session_name(saveid) # usually correct, but might be wrong if running overnight!

        # Stop recording
        time.sleep(1) # Need to wait for a bit since the recording system has some latency and we don't want to stop prematurely
        try:
            self._ecube_cleanup(saveid)
        except Exception as e:
            print(e)
            traceback.print_exc()
            log_exception(e)
            print('\n\neCube recording could not be stopped! Please manually stop the recording\n\n')
            return False

        # Save file to database
        print("Saving ecube file to database...")
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        suffix = '' # note: database functions don't take keyword arguements like custom_suffix=suffix
        filepath = os.path.join(ecube_path, ecube_session)
        if dbname == 'default':
            database.save_data(filepath, "ecube", saveid, False, False, suffix) # make sure to add the ecube datasource!
        else:
            database.save_data(filepath, "ecube", saveid, False, False, suffix, dbname=dbname)
        return super_result

    def _ecube_cleanup(self, saveid):
        from riglib.ecube import pyeCubeStream
        ec = pyeCubeStream.eCubeStream()
        active = ec.listremotesessions()
        stopped = saveid is None
        for session in active:
            if str(saveid) in session:
                ecube_session = session
                ec.remotestopsave(session)
                print('Stopped eCube recording session ' + session)
                log_str("Stopped recording " + session)
                stopped = True

        # Remove headstage sources so they can be added again later
        sources = ec.listadded()
        if len(sources[0]) > 0:
            for hs in np.unique(sources[0][0]):
                ec.remove(('Headstages', int(hs)))

        # Check that everything worked out
        if not stopped:
            raise Exception('Could not find active session for ' + ecube_session)

    
    @classmethod
    def pre_init(cls, saveid=None, record_headstage=False, headstage_connector=None, headstage_channels=None, **kwargs):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        cls.ecube_status = None
        if saveid is not None:
            from riglib.ecube import pyeCubeStream
            session_name = make_ecube_session_name(saveid)
            log_str("\n\nNew recording for task entry {}: {}".format(saveid, session_name))
            try:
                ec = pyeCubeStream.eCubeStream()
                ec.add(('AnalogPanel',))
                ec.add(('DigitalPanel',))
                if record_headstage:
                    ec.add(('Headstages', headstage_connector, headstage_channels))
                ec.remotesave(session_name)
                active_sessions = ec.listremotesessions()
                if session_name in active_sessions:
                    cls.ecube_status = "recording"
                else:
                    cls.ecube_status = "Could not start eCube recording. Make sure servernode is running!\n"
            except Exception as e:
                print(e)
                traceback.print_exc()
                log_exception(e)
                cls.ecube_status = e
            
        else:
            # Just a test, try to connect to servernode to make sure it's working
            from riglib.ecube import pyeCubeStream
            session_name = make_ecube_session_name(saveid)
            log_str("\n\nNew recording for task entry {}: {}".format(saveid, session_name))
            try:
                ec = pyeCubeStream.eCubeStream()
                cls.ecube_status = "testing"
            except Exception as e:
                print(e)
                traceback.print_exc()
                log_exception(e)
                cls.ecube_status = "Could not connect to eCube. Make sure servernode is running!\n"
        super().pre_init(saveid=saveid, **kwargs)

    def run(self):
        if not self.ecube_status in ["testing", "recording"]:
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.ecube_status)
            self.termination_err.seek(0)
            self.state = None
        try:
            super().run()
        except Exception as e:
            try:
                self._ecube_cleanup(self.saveid)
            except Exception as ee:
                log_str("Tried to cleanup ecube recording but couldn't...")
                log_exception(ee)
            finally:
                raise e

class EcubeFileData(CorticalData, traits.HasTraits):
    '''
    Streams data from an ecube file into BMI3D as neurondata
    '''

    ecube_bmi_filename = traits.String("", desc="File to playback in BMI")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Query the file to see the channels
        file = ecube.file.parse_file(self.ecube_bmi_filename)
        self.cortical_channels = file.channels

        # These get read by CorticalData when initializing the extractor
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            send_data_to_sink_manager=self.send_data_to_sink_manager, 
            channels=self.cortical_channels,
            ecube_bmi_filename=self.ecube_bmi_filename)
        self._neural_src_system_type = ecube.File
        
    @property 
    def sys_module(self):
        return ecube    

class EcubeData(CorticalData, traits.HasTraits):
    '''
    Streams neural data using ecube as the datasource. Use this if you want online neural data
    but don't want to use a decoder.
    '''

    streaming_channels = traits.Tuple((1,1), desc="Range of channels to stream")

    def init(self):
        self.cortical_channels = list(range(*self.streaming_channels))
        super().init()

    @property 
    def sys_module(self):
        return ecube   

class EcubeFileBMI(EcubeFileData, CorticalBMI):
    '''
    Streams data from an ecube file into a BMI decoder
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # These get read by CorticalData when initializing the extractor
        self._neural_src_type = source.MultiChanDataSource
        self._neural_src_kwargs = dict(
            send_data_to_sink_manager=self.send_data_to_sink_manager, 
            channels=self.decoder.units[:,0],
            ecube_bmi_filename=self.ecube_bmi_filename)
        self._neural_src_system_type = ecube.File

class EcubeBMI(CorticalBMI):
    '''
    BMI using ecube as the datasource.
    '''

    bmi_ecube_headstage = traits.Int(7, desc="Which headstage to use for BMI data")

    def init(self):
        self.neural_src_kwargs = dict(headstage=self.bmi_ecube_headstage)
        super().init()

    @property 
    def sys_module(self):
        return ecube   


class TestExperiment():
    state = None
    def __init__(self, *args, **kwargs):
        pass

    def pre_init(saveid, **kwargs):
        pass

    def start(self):
        import time
        time.sleep(1)

    def cleanup(self, database, saveid, **kwargs):
        pass

    def terminate(self):
        pass

    def join(self):
        pass

class TestClass(RecordECube, TestExperiment):
    pass

if __name__ == "__main__":

    TestClass.pre_init(saveid=10)
    exp = TestClass()
    exp.start()
    exp.cleanup(None, 10)
    # from riglib import experiment
    # proc = experiment.task_wrapper.TaskWrapper(
    #     subj=None, params=dict(), target_class=TestClass, saveid=10)
    # proc.start()