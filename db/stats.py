# cd code/bmi3d/db
# python manage.py shell

def statsget(sessionid):
    from riglib.experiment import report
    from tracker.models import TaskEntry
    d = report(TaskEntry.objects.get(pk=sessionid).get())
    depth = float(len(d['trials']['depth']['correct'])) / len(d['trials']['depth']['correct'] + d['trials']['depth']['incorrect'])
    flat = float(len(d['trials']['flat']['correct'])) / len(d['trials']['flat']['correct'] + d['trials']['flat']['incorrect'])
    return depth, flat
