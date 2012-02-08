import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

class ExpRun(object):
    def __init__(self, db):
        self.state = None
        self.experiment = None
        self.db = xmlrpclib.ServerProxy("http://localhost:8001/xmlrpc/")
    
    def run(self, data, save=True):
        name = self.db.get_name(data['task_id'])
        seq = self.sequence(data['sequence'])

    
    def sequence(self, data):
        if isinstance(data, dict):
            #parse the text input, save a new sequence object
            params = dict([(k, json.loads(v)) for k, v in data['params'].items()])
            seqdb = dict(
                generator_id=data['generator'], 
                task_id=taskid,
                name=data['name'], 
                params=params,
                static=data['static'])
            self._pipe_disp.send((("seqmake", "seqsave")[save], seqdb))
        else:
            self._pipe_disp.send(("seqget", int(data)))
        
        seq = self._pipe_disp.recv()
        if "seq" in seq:
            return seq['id'], experiment.generate.runseq, dict(seq=seq['seq'])
        elif seq['static']:
            gen = experiment.genlist[seq['gen']](**seq['params'])
            return seq['id'], experiment.generate.runseq, dict(seq=gen)
        else:
            return seq['id'], experiment.genlist[seq['gen']], seq['params']

