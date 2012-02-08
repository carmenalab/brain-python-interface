from SimpleXMLRPCServer import SimpleXMLRPCDispatcher

from django.http import HttpResponse
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject

def rpc_handler(request):
        response = HttpResponse(mimetype="application/xml")
        response.write(dispatcher._marshaled_dispatch(request.raw_post_data))
        return response

def get_taskname(idx):
    return Task.objects.get(pk=idx).name

def seq_get(data):
    if cmd == "seqget":
        seqdb = Sequence.objects.get(pk=data)
    else:
        static = data['static']
        del data['static']
        data['params'] = json.dumps(data['params'])
        seqdb = Sequence(**data)
        if static:
            gen = experiment.genlist[seqdb.generator.name]
            seqdb.sequence = cPickle.dumps(gen(**json.loads(seqdb.params)))
        if cmd == "seqsave":
            seqdb.save()
    
    if seqdb.sequence != '':
        self.pipe_disp.send(dict(id=seqdb.id, seq=cPickle.loads(seqdb.sequence)))
    else:
        self.pipe_disp.send(dict(id=seqdb.id, 
            gen=seqdb.generator.name, 
            params=json.loads(seqdb.params), 
            static=seqdb.generator.static))

def save_seq(data):
    pass


# you have to manually register all functions that are xml-rpc-able with the dispatcher
# the dispatcher then maps the args down.
# The first argument is the actual method, the second is what to call it from the XML-RPC side...
dispatcher.register_function(get_taskname, 'get_taskname')