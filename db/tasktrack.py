import os
import sys
import json
import time
import cPickle
import threading
import xmlrpclib
import multiprocessing as mp

import numpy as np

from riglib import experiment
from riglib.experiment import features
from tracker import models
import websocket

from json_param import Parameters

class Track(object):
    def __init__(self):
        self.status = mp.Array('c', 256)
        self.task = None
        self.proc = None
        self.websock = websocket.Server(self.notify)
        self.cmds, self._cmds = mp.Pipe()

    def notify(self, msg):
        if msg['status'] == "error" or msg['state'] == "stopped":
            self.status.value = ""

    def runtask(self, **kwargs):
        self.status.value = "testing" if 'saveid' in kwargs else "running"
        self.task = ObjProxy(self.cmds)
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
    sys.stdout = websock
    os.nice(0)
    status = "running" if 'saveid' in kwargs else "testing"
    class NotifyFeat(object):
        def set_state(self, state, *args, **kwargs):
            l = time.time() - self.event_log[0][2] if len(self.event_log) > 0 else 0
            rep = dict(status=status, state=state or "stopped", length=l)
            if state == "wait":
                rep.update(report.general(self.__class__, self.event_log))
            websock.send(rep)
            super(NotifyFeat, self).set_state(state, *args, **kwargs)

        def run(self):
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
                
        task.cleanup()
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        websock.send(dict(status="error", msg=err.read()))
        err.seek(0)
        print err.read()
    
    print "****************Exit task proc"

class Task(object):
    def __init__(self, subj, task, feats, params, seq=None, saveid=None):
        self.saveid = saveid
        self.taskname = task.name
        self.subj = subj
        self.params = Parameters(params)
        if self.saveid is not None:
            try:
                import comedi
                self.com = comedi.comedi_open("/dev/comedi0")
                comedi.comedi_dio_bitfield2(self.com,0,16,0,16)
            except:
                print "No comedi, cannot start"
        
        Exp = experiment.make(task.get(), feats=feats)
        self.params.trait_norm(Exp.class_traits())
        if issubclass(Exp, experiment.Sequence):
            gen, gp = seq.get()
            sequence = gen(Exp, **gp)
            exp = Exp(sequence, **self.params.params)
        else:
            exp = Exp(**self.params.params)
        
        exp.start()
        self.task = exp

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
        database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
        if self.saveid is not None:
            try:
                import comedi
                comedi.comedi_dio_bitfield2(self.com, 0, 16, 16, 16)
            except:
                pass
            database.save_log(self.saveid, self.task.event_log)
            time.sleep(2) #Give plexon a chance to catch up
            if "calibration" in self.taskname:
                caltype = self.task.calibration.__class__.__name__
                params = Parameters.from_dict(self.task.calibration.__dict__)
                if hasattr(self.task.calibration, '__getstate__'):
                    params = Parameters.from_dict(self.task.calibration.__getstate__())
                database.save_cal(self.subj, self.task.calibration.system,
                    caltype, params.to_json())
            
            if issubclass(self.task.__class__, features.EyeData):
                database.save_data(self.task.eyefile, "eyetracker", self.saveid)
            
            if issubclass(self.task.__class__, features.SaveHDF):
                database.save_data(self.task.h5file.name, "hdf", self.saveid)

            if issubclass(self.task.__class__, features.RelayPlexon):
                if self.task.plexfile is not None:
                    database.save_data(self.task.plexfile, "plexon", self.saveid, True, False)

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
