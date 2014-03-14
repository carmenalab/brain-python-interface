import os
#os.nice(4)

try:
    import models
    subjects = models.Subject.objects.all()
    if len(subjects) == 0:
        subj = models.Subject(name='testing')
        subj.save()

    for m in [models.Task, models.Feature, models.Generator, models.System]:
        m.populate()
except:
    import traceback
    traceback.print_exc()
