'''
Database initialization code. When 'db.tracker' is imported, it goes through the database and ensures that 
	1) at least one subject is present
	2) all the tasks from 'tasklist' appear in the db
	3) all the features from 'featurelist' appear in the db
	4) all the generators from all the tasks appear in the db 
'''

import os
#os.nice(4)

try:
    from . import models
    subjects = models.Subject.objects.all()
    if len(subjects) == 0:
        subj = models.Subject(name='testing')
        subj.save()

    for m in [models.Task, models.Feature, models.Generator, models.System]:
        m.populate()
except:
    import traceback
    traceback.print_exc()
