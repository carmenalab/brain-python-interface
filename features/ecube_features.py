import time
import os
from datetime import datetime
from riglib.experiment import traits
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
        from riglib.ecube import pyeCubeStream
        ecube_session = make_ecube_session_name(saveid) # usually correct, but might be wrong if running overnight!

        # Stop recording
        time.sleep(1) # Need to wait for a bit since the recording system has some latency and we don't want to stop prematurely
        stopped = False
        try:
            ec = pyeCubeStream.eCubeStream(debug=True)
            active = ec.listremotesessions()
            for session in active:
                if str(saveid) in session:
                    ecube_session = session
                    ec.remotestopsave(session)
                    print('Stopped eCube recording session ' + session)
                    log_str("Stopped recording " + session)
                    stopped = True

            if not stopped:
                raise Exception('Could not find active session for ' + ecube_session)
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

    
    @classmethod
    def pre_init(cls, saveid=None, record_headstage=False, headstage_connector=None, headstage_channels=None, **kwargs):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        cls.ecube_status = None
        if saveid is not None:
            from riglib.ecube import pyeCubeStream
            session_name = make_ecube_session_name(saveid)
            log_str("New recording for task entry {}: {}".format(saveid, session_name))
            try:
                ec = pyeCubeStream.eCubeStream(debug = True)
                ec.add(('AnalogPanel',))
                ec.add(('DigitalPanel',))
                if record_headstage:
                    ec.add(('Headstages', headstage_connector, headstage_channels))
                ec.remotesave(session_name)
                active_sessions = ec.listremotesessions()
            except Exception as e:
                print(e)
                traceback.print_exc()
                log_exception(e)
                active_sessions = []
            if session_name in active_sessions:
                cls.ecube_status = "recording"
            else:
                cls.ecube_status = "Could not start eCube recording. Make sure servernode is running!"
        else:
            cls.ecube_status = "testing"
        super().pre_init(saveid=saveid, **kwargs)

    def run(self):
        if not self.ecube_status in ["testing", "recording"]:
            raise ConnectionError(self.ecube_status)
        super().run()

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