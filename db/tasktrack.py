'''Needs docs'''


import os
import sys
import time
import xmlrpclib
import multiprocessing as mp
import collections


from riglib import experiment
import websocket

from json_param import Parameters

class Track(object):
    '''
    Tracker for task instantiation running in a separate process
    '''
    def __init__(self):
        # shared memory to store the status of the task in a char array
        self.status = mp.Array('c', 256)
        self.task = None
        self.proc = None
        self.websock = websocket.Server(self.notify)
        self.cmds, self._cmds = mp.Pipe()

    def notify(self, msg):
        if msg['status'] == "error" or msg['State'] == "stopped":
            self.status.value = ""

    def runtask(self, **kwargs):
        '''
        Begin running of task
        '''
        # initialize task status
        self.status.value = "testing" if 'saveid' in kwargs else "running"

        # create a proxy for interacting with attributes/functions of the task.
        # The task runs in a separate process and we cannot directly access python 
        # attributes of objects in other processes
        self.task = ObjProxy(self.cmds)

        # Spawn the process
        args = (self.cmds, self._cmds, self.websock)
        self.proc = mp.Process(target=runtask, args=args, kwargs=kwargs)
        self.proc.start()
        
    def __del__(self):
        self.websock.stop()

    def pausetask(self):
        self.status.vaue = self.task.pause()

    def stoptask(self):
        assert self.status.value in "testing,running"
        try:
            self.task.end_task()
        except Exception as e:
            import cStringIO
            import traceback
            err = cStringIO.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            return dict(status="error", msg=err.read())

        status = self.status.value
        self.status.value = ""
        self.task = None
        return status

def runtask(cmds, _cmds, websock, **kwargs):
    import time
    from riglib.experiment import report

    # Rerout prints to stdout to the websocket
    sys.stdout = websock

    # os.nice sets the 'niceness' of the task, i.e. how willing the process is
    # to share resources with other OS processes. Zero is neutral
    os.nice(0)

    status = "running" if 'saveid' in kwargs else "testing"
    class NotifyFeat(object):
        def set_state(self, state, *args, **kwargs):
        
            l = time.time() - self.event_log[0][2] if len(self.event_log) > 0 else 0
            rep = dict(status=status, State=state or "stopped", length=l)
            rep.update(report.general(self.__class__, self.event_log, self.reportstats, self.ntrials, self.nrewards, self.reward_len))
            rep['length'] = time.time() - self.task_start_time

            self.reportstats['status'] = status
            self.reportstats['State'] = state or 'stopped'
            
            websock.send(rep)
            #websock.send(self.reportstats)
            super(NotifyFeat, self).set_state(state, *args, **kwargs)

        def run(self):
            f = open('/home/helene/code/bmi3d/log/ajax_task_startup', 'a')
            f.write('trying to execute NotifyFeat.run()\n')
            try:
                super(NotifyFeat, self).run()
            except:
                import cStringIO
                import traceback
                err = cStringIO.StringIO()
                traceback.print_exc(None, err)
                err.seek(0)
                websock.send(dict(status="error", msg=err.read()))
            finally:
                cmds.send(None)
            f.close()

    # Force all tasks to use the Notify feature defined above
    kwargs['feats'].insert(0, NotifyFeat)

    try:
        task = Task(**kwargs)
        cmd = _cmds.recv()
        while cmd is not None and task.task.state is not None:
            try:
                ret = getattr(task, cmd[0])(*cmd[1], **cmd[2])
                _cmds.send(ret)
                cmd = _cmds.recv()
            except KeyboardInterrupt:
                cmd = None
            except Exception as e:
                _cmds.send(e)
                cmd = _cmds.recv()
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        with open('/tmp/exceptions', 'w') as fp:
            err.seek(0)
            fp.write(err.read())
        err.seek(0)
        websock.send(dict(status="error", msg=err.read()))
        err.seek(0)
        print err.read()
    sys.stdout = sys.__stdout__
    task.cleanup()
    print "****************Exit task proc"

class Task(object):
    def __init__(self, subj, task, feats, params, seq=None, saveid=None):
        '''
        Parameters
        ----------
        subj : database record for subject
        task : database record for task
        feats : list of features to enable for the task
        params : user input on configurable task parameters
        '''
        f = open('/home/helene/code/bmi3d/log/ajax_task_startup', 'a')
        f.write('tasktrack.Task.__init__\n')
        self.saveid = saveid
        self.taskname = task.name
        self.subj = subj
        self.params = Parameters(params)

        # Send pulse to plexon box to start saving to file?
        if self.saveid is not None:
            try:
                import comedi
                self.com = comedi.comedi_open("/dev/comedi0")
                comedi.comedi_dio_bitfield2(self.com,0,16,0,16)
                time.sleep(2)
            except:
                print "No comedi, cannot start"
        
        base_class = task.get()
        f.write('Created base class: %s\n' % base_class)
        Exp = experiment.make(base_class, feats=feats)
        self.params.trait_norm(Exp.class_traits())
        if issubclass(Exp, experiment.Sequence):
            gen, gp = seq.get()
            sequence = gen(Exp, **gp)
            exp = Exp(sequence, **self.params.params)
        else:
            exp = Exp(**self.params.params)
        
        exp.start()
        self.task = exp
        f.close()

    def report(self):
        return experiment.report(self.task)
    
    def pause(self):
        self.task.pause = not self.task.pause
        return "pause" if self.task.pause else "running"
    
    def end_task(self):
        self.task.end_task()
    
    def get_state(self):
        return self.task.state

    def __getattr__(self, attr):
        return getattr(self, attr)
    
    def cleanup(self):
        self.task.join()
        print "Calling saveout/task cleanup code"
        if self.saveid is not None:
            try:
                import comedi
                comedi.comedi_dio_bitfield2(self.com, 0, 16, 16, 16)
            except:
                pass

            database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
            self.task.cleanup(database, self.saveid, subject=self.subj)

class ObjProxy(object):
    def __init__(self, cmds):
        self.cmds = cmds

    def __getattr__(self, attr):
        self.cmds.send(("__getattr__", [attr], {}))
        ret = self.cmds.recv()
        if isinstance(ret, Exception):
            return FuncProxy(attr, self.cmds)

        return ret

class FuncProxy(object):
    def __init__(self, func, pipe):
        self.pipe = pipe
        self.cmd = func
    def __call__(self, *args, **kwargs):
        self.pipe.send((self.cmd, args, kwargs))
        return self.pipe.recv()
