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

sockets = []

class ClientSocket(websocket.WebSocketHandler):
    def open(self):
        sockets.append(self)
        print "WebSocket opened"

    def on_close(self):
        print "WebSocket closed"
        sockets.remove(self)

    def check_origin(self):
        return True

class Server(mp.Process):
    def __init__(self, notify=None):
        super(self.__class__, self).__init__()
        self._pipe, self.pipe = os.pipe()
        self._outp, self.outp = os.pipe()
        self.outqueue = ""
        self.notify = notify
        self.start()

    def run(self):
        print "Running websocket service"
        application = tornado.web.Application([
            (r"/connect", ClientSocket),
        ])

        application.listen(8001)
        self.ioloop = tornado.ioloop.IOLoop.instance()
        self.ioloop.add_handler(self._pipe, self._send, self.ioloop.READ)
        self.ioloop.add_handler(self._outp, self._stdout, self.ioloop.READ)
        self.ioloop.start()

    def _stdout(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        for sock in sockets:
            sock.write_message(msg)

    def _send(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        if msg == "stop":
            self.ioloop.stop()
        else:
            for sock in sockets:
                sock.write_message(msg)
    
    def send(self, msg):
        if self.notify is not None:
            self.notify(msg)
        if not isinstance(msg, (str, unicode)):
            msg = json.dumps(msg)

        os.write(self.pipe, struct.pack('I', len(msg))+msg)

    def write(self, data):
        '''Used for stdout hooking'''
        self.outqueue += data
        self.flush()

    def flush(self):
        msg = json.dumps(dict(status="stdout", msg=cgi.escape(self.outqueue)))
        os.write(self.outp, struct.pack('I', len(msg))+msg)
        self.outqueue = ""

    def stop(self):
        print "Stopping websocket service"
        self.send("stop")

def test():
    serv = Server()
    serv.send(dict(state="working well"))
    serv.stop()

if __name__ == "__main__":
    test()