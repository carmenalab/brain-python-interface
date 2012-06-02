import os
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

class Server(mp.Process):
    def __init__(self, notify=None):
        super(self.__class__, self).__init__()
        self._pipe, self.pipe = os.pipe()
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
        self.ioloop.start()

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

    def stop(self):
        print "Stopping websocket service"
        self.send("stop")

def test():
    serv = Server()
    serv.send(dict(state="working well"))
    serv.stop()

if __name__ == "__main__":
    test()