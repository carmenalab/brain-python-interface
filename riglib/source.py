'''
Generic data source module. Sources run in separate processes and continuously collect/save
data and interact with the main process through the methods here.
'''

import os
import sys
import time
import inspect
import traceback
import multiprocessing as mp
from multiprocessing import sharedctypes as shm
import ctypes

import numpy as np

import sink
from . import FuncProxy


class DataSource(mp.Process):
    ''' Docstring '''
    def __init__(self, source, bufferlen=10, name=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(DataSource, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = source.__module__.split('.')[-1]
        self.filter = None
        self.source = source
        self.source_kwargs = kwargs
        self.bufferlen = bufferlen
        self.max_len = bufferlen * int(self.source.update_freq)
        self.slice_size = self.source.dtype.itemsize
        
        self.lock = mp.Lock()
        self.idx = shm.RawValue('l', 0)
        self.data = shm.RawArray('c', self.max_len * self.slice_size)
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)
        self.stream = mp.Event()
        self.last_idx = 0

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

    def start(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.sinks = sink.sinks
        super(DataSource, self).start(*args, **kwargs)

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        try:
            system = self.source(**self.source_kwargs)
            system.start()
        except Exception as e:
            print e
            self.status.value = -1

        streaming = True
        size = self.slice_size
        while self.status.value > 0:
            f = open(os.path.join(os.getenv("HOME"), 'code/bmi3d/log/source'), 'a')
            f.write("1\n")
            f.close()

            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    if cmd == "getattr":
                        ret = getattr(system, args[0])
                    else:
                        ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    ret = e
                self.lock.release()
                self._pipe.send(ret)
                self.cmd_event.clear()
            if self.stream.is_set():
                self.stream.clear()
                streaming = not streaming
                if streaming:
                    self.idx.value = 0
                    system.start()
                else:
                    system.stop()
            if streaming:
                data = system.get()
                self.sinks.send(self.name, data)
                if data is not None:
                    try:
                        self.lock.acquire()
                        i = self.idx.value % self.max_len
                        self.data[i*size:(i+1)*size] = np.array(data).tostring()
                        self.idx.value += 1
                        self.lock.release()
                    except Exception as e:
                        print e
            else:
                time.sleep(.001)
        system.stop()

    def get(self, all=False, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.status.value <= 0:
            raise Exception('Error starting datasource ' + self.name)
            
        self.lock.acquire()
        i = (self.idx.value % self.max_len) * self.slice_size
        if all:
            if self.idx.value < self.max_len:
                data = self.data[:i]
            else:
                data = self.data[i:]+self.data[:i]
        else:
            mlen = min((self.idx.value - self.last_idx), self.max_len)
            last = ((self.idx.value - mlen) % self.max_len) * self.slice_size
            if last > i:
                data = self.data[last:] + self.data[:i]
            else:
                data = self.data[last:i]
            
        self.last_idx = self.idx.value
        self.lock.release()
        try:
            data = np.fromstring(data, dtype=self.source.dtype)
        except:
            print "can't get fromstring..."

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    # TODO -- change name, add documentation
    def read(self, n_pts=1, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.status.value <= 0:
            raise Exception('Error starting datasource ' + self.name)
            
        self.lock.acquire()
        idx = self.idx.value % self.max_len 
        i = idx * self.slice_size
        
        if n_pts > self.max_len:
            n_pts = self.max_len

        if idx >= n_pts:  # no wrap-around required
            data = self.data[(idx-n_pts)*self.slice_size:idx*self.slice_size]
        else:
            data = self.data[-(n_pts-idx)*self.slice_size:] + self.data[:idx*self.slice_size]

        self.lock.release()
        try:
            data = np.fromstring(data, dtype=self.source.dtype)
        except:
            print "can't get fromstring..."

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def pause(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.stream.set()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.status.value = -1
    
    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.stop()

    def __getattr__(self, attr):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        elif not attr.beginsWith("__"):
            print "getting attribute %s"%attr
            self.pipe.send(("getattr", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()
        raise AttributeError(attr)


class MultiChanDataSource(mp.Process):
    '''
    Multi-channel version of 'DataSource'
    '''
    def __init__(self, source, bufferlen=1, **kwargs):
        '''
        Constructor for MultiChanDataSource

        Parameters
        ----------
        source: class 
            lower-level class for interacting directly with the incoming data (e.g., plexnet)
        bufferlen: int
            Constrains the maximum amount of data history stored by the source
        kwargs: dict, optional, default = {}
            For the multi-channel data source, you MUST specify a 'channels' keyword argument
            Note that kwargs['channels'] does not need to a list of integers,
            it can also be a list of strings (e.g., see feedback multi-chan data source for IsMore).
        '''
        super(MultiChanDataSource, self).__init__()
        self.name = source.__module__.split('.')[-1]
        self.filter = None
        self.source = source
        self.source_kwargs = kwargs
        self.bufferlen = bufferlen
        self.max_len = int(bufferlen * self.source.update_freq)
        
        self.chan_to_row = dict()
        for row, chan in enumerate(kwargs['channels']):
            self.chan_to_row[chan] = row
        self.n_chan = len(kwargs['channels'])

        dtype = self.source.dtype  # e.g., np.dtype('float') for LFP
        self.slice_size = dtype.itemsize
        self.idxs = shm.RawArray('l', self.n_chan)
        self.last_read_idxs = np.zeros(self.n_chan)
        rawarray = shm.RawArray('c', self.n_chan * self.max_len * self.slice_size)
        self.data = np.frombuffer(rawarray, dtype).reshape((self.n_chan, self.max_len))
        
        self.lock = mp.Lock()
        self.pipe, self._pipe = mp.Pipe()
        self.cmd_event = mp.Event()
        self.status = mp.Value('b', 1)
        self.stream = mp.Event()

        self.methods = set(n for n in dir(source) if inspect.ismethod(getattr(source, n)))

    def start(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.sinks = sink.sinks
        super(MultiChanDataSource, self).start(*args, **kwargs)

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "Starting datasource %r" % self.source
        try:
            system = self.source(**self.source_kwargs)
            system.start()
        except Exception as e:
            print e
            self.status.value = -1

        streaming = True
        size = self.slice_size
        while self.status.value > 0:
            if self.cmd_event.is_set():
                cmd, args, kwargs = self._pipe.recv()
                self.lock.acquire()
                try:
                    if cmd == "getattr":
                        ret = getattr(system, args[0])
                    else:
                        ret = getattr(system, cmd)(*args, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    ret = e
                self.lock.release()
                self._pipe.send(ret)
                self.cmd_event.clear()
            if self.stream.is_set():
                self.stream.clear()
                streaming = not streaming
                if streaming:
                    self.idx.value = 0
                    system.start()
                else:
                    system.stop()
            if streaming:
                # system.get() must return a tuple (chan, data), where: 
                #   chan is the the channel number
                #   data is a numpy array with a dtype (or subdtype) of
                #   self.source.dtype
                chan, data = system.get()
                # for now, assume no multi-channel data source is registered
                # TODO -- how to send MCDS data to a sink? (problem is that
                #    "data" has a variable length each time and has no 
                #    fixed-size dtype, which sinks require)
                # self.sinks.send(self.name, data)
                if data is not None:
                    try:
                        self.lock.acquire()

                        try:
                            row = self.chan_to_row[chan]  # row in ringbuffer corresponding to this channel
                        except KeyError:
                            # print 'data source was not configured to get data on channel', chan
                            pass
                        else:
                            n_pts = len(data)
                            max_len = self.max_len

                            if n_pts > max_len:
                                data = data[-max_len:]
                                n_pts = max_len

                            idx = self.idxs[row] # for this channel, idx in ringbuffer
                            if idx + n_pts <= self.max_len:
                                self.data[row, idx:idx+n_pts] = data
                                idx = (idx + n_pts)
                                if idx >= max_len:
                                    idx = idx % max_len
                            else: # need to write data at both end and start of buffer
                                self.data[row, idx:] = data[:max_len-idx]
                                self.data[row, :n_pts-(max_len-idx)] = data[max_len-idx:]
                                idx = n_pts-(max_len-idx)
                            self.idxs[row] = idx

                        self.lock.release()
                    except Exception as e:
                        print e
            else:
                time.sleep(.001)
        system.stop()
        print "ended datasource %r"%self.source

    def get(self, n_pts, channels, **kwargs):
        '''
        Return the most recent n_pts of data from the specified channels.

        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.status.value <= 0:
            raise Exception('Error starting datasource ' + self.name)

        self.lock.acquire()
        
        # these channels must be a subset of the channels passed into __init__
        n_chan = len(channels)
        data = np.zeros((n_chan, n_pts), dtype=self.source.dtype)

        if n_pts > self.max_len:
            n_pts = self.max_len

        for chan_num, chan in enumerate(channels):
            try:
                row = self.chan_to_row[chan]
            except KeyError:
                print 'data source was not configured to get data on channel', chan
            else:  # executed if try clause does not raise a KeyError
                idx = self.idxs[row]
                if idx >= n_pts:  # no wrap-around required
                    data[chan_num, :] = self.data[row, idx-n_pts:idx]
                else:
                    data[chan_num, :n_pts-idx] = self.data[row, -(n_pts-idx):]
                    data[chan_num, n_pts-idx:] = self.data[row, :idx]
                self.last_read_idxs[row] = idx

        self.lock.release()

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def get_new(self, channels, **kwargs):
        '''
        Return the new (unread) data from the specified channels.
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if self.status.value <= 0:
            raise Exception('Error starting datasource ' + self.name)

        self.lock.acquire()
        
        # these channels must be a subset of the channels passed into __init__
        n_chan = len(channels)
        data = []

        for chan in channels:
            try:
                row = self.chan_to_row[chan]
            except KeyError:
                print 'data source was not configured to get data on channel', chan
                data.append(None)
            else:  # executed if try clause does not raise a KeyError
                idx = self.idxs[row]
                last_read_idx = self.last_read_idxs[row]
                if last_read_idx <= idx:  # no wrap-around required
                    data.append(self.data[row, last_read_idx:idx])
                else:
                    data.append(np.hstack((self.data[row, last_read_idx:], self.data[row, :idx])))
            self.last_read_idxs[row] = idx

        self.lock.release()

        if self.filter is not None:
            return self.filter(data, **kwargs)
        return data

    def pause(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.stream.set()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.status.value = -1
    
    def __del__(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.stop()

    def __getattr__(self, attr):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if attr in self.methods:
            return FuncProxy(attr, self.pipe, self.cmd_event)
        elif not attr.beginsWith("__"):
            print "getting attribute %s" % attr
            self.pipe.send(("getattr", (attr,), {}))
            self.cmd_event.set()
            return self.pipe.recv()
        raise AttributeError(attr)


if __name__ == "__main__":
    from riglib import motiontracker
    sim = DataSource(motiontracker.make_simulate(8))
    sim.start()
    #sim.get()
