import multiprocessing as mp
from multiprocessing import sharedctypes as shm

import numpy as np

#Update frequency for each modality, for calculating size of shm
update_freq = dict(
    eyetracker=500,
    motion=60,
    neuron=2048,
)
#Size of a single time slice for each modality
size = dict(
    eyetracker=2,
    motion=32,
    neuron=256,
)

class MemTrack(object):
    def __init__(self, eyebuf=10, motionbuf=10):
        esize = eyebuf*update_freq['eyetracker']*size['eyetracker']
        msize = motionbuf*update_freq['motion']*size['motion']
        self.size = size
        self.msize = dict(
            eyetracker=eyebuf*update_freq['eyetracker'],
            motion=motionbuf*update_freq['motion']
        )
        print "Allocating e=%s, m=%s"%(esize, msize)
        self.data = dict(
            eyetracker = shm.RawArray('d', esize),
            motion = shm.RawArray('d', msize)
        )
        self.idx = dict(
            eyetracker = shm.RawValue('i', -1),
            motion = shm.RawValue('i', -1),
        )
        self.locks = dict(eyetracker=mp.Lock(), motion=mp.Lock())
        self.funcs = dict(
            eyetracker = self._update_eyetracker,
            motion = self._update_motion,
        )
        self.procs = dict()
        self.proxy = dict(
            eyetracker=ObjProxy(),
        )
    
    def start(self, modalities=None):
        if modalities is None:
            modalities = ['eyetracker', 'motion']
        elif not isinstance(modalities, (list, tuple)):
            modalities = [modalities]
        
        for mode in modalities:
            self.idx[mode].value = 1
            args = (self.locks[mode], self.data[mode], self.idx[mode])
            proc = mp.Process(target=self.funcs[mode], args=args)
            self.procs[mode] = proc
            proc.start()

    def stopall(self):
        for mode in self.procs.keys():
            #This should force all modalities to exit after the next refresh
            self.idx[mode].value = -1
    
    def _update_eyetracker(self, lock, data, idx):
        '''Must be run as a separate process, to update the eyetracker data'''
        import eyetracker
        system = eyetracker.System()
        system.start()
        size = self.size['eyetracker']
        msize = self.msize['eyetracker']
        proxy = self.proxy['eyetracker']
        while idx.value > 0:
            try:
                func = proxy.cmd.get_nowait()
                proxy._pipe.send(getattr(system, func[0])(*func[1], **func[2]))
            except:
                pass
            xy = system.get()
            if xy is not None:
                lock.acquire()
                i = idx.value % msize
                data[i*size:(i+1)*size] = xy
                idx.value += 1
                lock.release()
    
    def get(self, mode):
        self.locks[mode].acquire()
        i = self.idx[mode].value % self.msize[mode]
        if self.idx[mode].value > self.msize[mode]:
            data = self.data[mode][i:]+self.data[mode][:i]
        else:
            data = self.data[mode][:i]
        self.idx[mode].value = 0
        self.locks[mode].release()
        try:
            return np.array(data).reshape(-1, size[mode])
        except:
            pass
    
    def flush(self):
        for mode in self.procs.keys():
            self.locks[mode].acquire()
            self.idx[mode].value = 0
            self.locks[mode].release()
    
    def _update_motion(self, lock, data, idx):
        pass
    
    def __del__(self):
        self.stopall()

class ObjProxy(object):
    def __init__(self):
        self.cmd = mp.Queue()
        self.pipe, self._pipe = mp.Pipe()
    
    def __getattr__(self, attr):
        return FuncProxy(attr, self.cmd, self._pipe)

class FuncProxy(object):
    def __init__(self, name, cmdqueue, pipe):
        self.cmd = cmdqueue
        self.pipe = pipe
        self.name = name

    def __call__(self, *args, **kwargs):
        self.cmd.put((self.name, args, kwargs))
        return self.pipe.recv()

if __name__ == "__main__":
    track = MemTrack()
    track.start("eyetracker")
