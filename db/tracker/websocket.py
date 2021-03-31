'''
An extension of Tornado's web socket which enables the task to
print data to the web interface while the task is running.
'''

import os
import sys
import cgi
import json
import struct
import multiprocessing as mp
import tornado.ioloop
import tornado.web
from tornado import websocket
import io, traceback
import copy

sockets = []

class ClientSocket(websocket.WebSocketHandler):
    def open(self):
        sockets.append(self)
        print("WebSocket opened")

    def on_close(self):
        print("WebSocket closed")
        sockets.remove(self)

    def check_origin(self, origin):
        '''
        Returns a boolean indicating whether the requesting URL is one that the
        handler will respond to. For this websocket, everyone with access gets a response since
        we're running the server locally (or over ssh tunnels) and not over the regular internet.

        Parameters
        ----------
        origin : string?
            The URL from which the request originated

        Returns
        -------
        boolean
            Returns True if the request originates from a valid URL

        See websocket.WebSocketHandler.check_origin for additional documentation
        '''
        return True


class Server(mp.Process):
    '''
    Spawn a process to deal with the websocket asynchronously, without halting other webserver operations.
    '''
    def __init__(self, notify=None):
        super(self.__class__, self).__init__()
        self._pipe, self.pipe = os.pipe()
        self._outp, self.outp = os.pipe()
        self.outqueue = ""
        self.notify = notify
        self.start()

    def run(self):
        '''
        Main function to run in the process. See mp.Process.run() for additional documentation.
        '''
        print("Running websocket service")
        application = tornado.web.Application([
            (r"/connect", ClientSocket),
        ])

        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())

        application.listen(8001)
        self.ioloop = tornado.ioloop.IOLoop.current()
        self.ioloop.add_handler(self._pipe, self._send, self.ioloop.READ)
        self.ioloop.add_handler(self._outp, self._stdout, self.ioloop.READ)
        self.ioloop.start()

    def send(self, msg):
        # notify the task tracker that data is being sent to the web interface
        if self.notify is not None:
            self.notify(msg)

        # force the message to string, if necessary (e.g., NotifyFeat.set_state sends a dict)
        if not isinstance(msg, str):
            msg = json.dumps(msg)

        # Write to 'self.pipe'. The write apparently triggers the function self._stdout to run
        os.write(self.pipe, struct.pack('I', len(msg)) + bytes(msg, 'utf8'))

    def _stdout(self, fd, event):
        '''
        Handler for self._pipe; Read the data from the input pipe and propagate the data to all the listening sockets
        '''
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)

        # write the message to all the sockets--could be multiple websockets open if multiple people are observing the same experiment
        for sock in sockets:
            # each 'sock' is assumed to be an instance of the ClientSocket above
            sock.write_message(msg)

    def stop(self):
        print("Stopping websocket service")
        self.send(dict(status="stopped", State="stopped"))
        self.join()

    ##### Currently unused functions below this line #####
    def write(self, data):
        '''Used for stdout hooking'''
        self.outqueue += data
        self.flush()

    def flush(self):
        msg = json.dumps(dict(status="stdout", msg=cgi.html.escape(self.outqueue)))
        os.write(self.outp, struct.pack('I', len(msg)) + bytes(msg, 'utf8'))
        self.outqueue = ""

    def _send(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        if msg == "stop":
            self.ioloop.stop()
        else:
            for sock in sockets:
                sock.write_message(msg)


class NotifyFeat(object):
    '''
    Send task report and state data to display on the web inteface
    '''
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.websock = kwargs.pop('websock')
        self.tracker_status = kwargs.pop('tracker_status')
        self.prev_stats = None

    def _cycle(self):
        super()._cycle()
        self.reportstats['status'] = str(self.tracker_status)
        if self.reportstats != self.prev_stats:
            self.websock.send(self.reportstats)
            self.prev_stats = copy.deepcopy(self.reportstats)

    def run(self):
        try:
            super().run()
        except:
            err = io.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            self.websock.send(dict(status="error", msg=err.read()))
        finally:
            if self.terminated_in_error:
                self.websock.send(dict(status="error", msg=self.termination_err.read()))
            else:
                self.reportstats['status'] = str(self.tracker_status)
                self.websock.send(self.reportstats)

    def print_to_terminal(self, *args):
        sys.stdout = sys.__stdout__
        super().print_to_terminal(*args)
        sys.stdout = self.websock


class WinNotifyFeat(object):
    '''
    Stop the task gracefully on windows without websockets
    '''
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker_end_of_pipe = kwargs.pop('tracker_end_of_pipe')
        self.tracker_status = kwargs.pop('tracker_status')

    def run(self):
        try:
            super().run()
        except:
            err = io.StringIO()
            traceback.print_exc(None, err)
            err.seek(0)
            print(err.read())
        finally:
            self.tracker_end_of_pipe.send(None)
