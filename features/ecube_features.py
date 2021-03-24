import time
from datetime import datetime

def make_ecube_session_name(saveid):
    now = datetime.now()
    session = now.strftime("%Y-%m-%d")
    number = str(saveid) if saveid else "Test"
    return session + "_BMI3D_te" + number

class RecordECube():

    def cleanup(self, database, saveid, **kwargs):
        '''
        Function to run at 'cleanup' time, after the FSM has finished executing. See riglib.experiment.Experiment.cleanup
        This 'cleanup' method remotely stops the plexon file recording and then links the file created to the database ID for the current TaskEntry
        '''
        super_result = super().cleanup(database, saveid, **kwargs)
        from riglib.ecube import pyeCubeStream
        ecube_session = make_ecube_session_name(saveid) # usually correct, but might be wrong if running overnight!

        # Stop recording
        try:
            ec = pyeCubeStream.eCubeStream()
            active = ec.listremotesessions()
            for session in active:
                if str(saveid) in session:
                    ecube_session = session
                    ec.remotestopsave(session)
                    print('Stopped eCube recording session ' + session)
        except Exception as e:
            print(e)
            print('\n\neCube recording could not be stopped! Please manually stop the recording\n\n')
            return False

        # Save file to database
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(ecube_session, "ecube", saveid, False, False, custom_suffix='') # make sure to add the ecube datasource!
        else:
            database.save_data(ecube_session, "ecube", saveid, False, False, custom_suffix='', dbname=dbname)
        return super_result

    
    @classmethod
    def pre_init(cls, saveid=None):
        '''
        Run prior to starting the task to remotely start recording from the plexon system
        '''
        if saveid is not None:
            from riglib.ecube import pyeCubeStream
            session_name = make_ecube_session_name(saveid)
            #try:
            ec = pyeCubeStream.eCubeStream(debug = True) #[('DigitalPanel',), ('AnalogPanel',)])
            ec.add(('AnalogPanel',))
            ec.add(('DigitalPanel',))
            ec.remotesave(session_name)
            active_sessions = ec.listremotesessions()
            #except Exception as e:
            #    print(e)
            #    active_sessions = []
            if session_name not in active_sessions:
                raise Exception("Could not start eCube recording. Make sure servernode is running!")
        super().pre_init(saveid=saveid)


class TestExperiment():
    state = None
    def __init__(self, *args, **kwargs):
        pass

    def pre_init(saveid):
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