import json
from models import TaskEntry

expqueue = []

class CommitFeat(object):
	def _start_None(self):
		super(CommitFeat, self)._start_None()
		te = TaskEntry.objects.get(pk=expqueue[0][0])
		te.report = json.dumps(expqueue[0][1].event_log)
		te.save()
		expqueue.pop()