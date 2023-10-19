'''
Utility to generate test data for aopy

Run `python manage.py flush` before using
'''

from db.boot_django import boot_django
boot_django()

# Make some test entries for subject, experimenter, and task        
from db.tracker import models
subj = models.Subject(name="test_subject")
subj.save()
expm = models.Experimenter(name="experimenter_1")
expm.save()
task = models.Task(name="manual control")
task.save()
task = models.Task(name="tracking")
task.save()
feat = models.Feature(name="feat_1")
feat.save()
gen = models.Generator(name="test_gen", static=False)
gen.save()
seq = models.Sequence(generator_id=gen.id, task_id=task.id, name="test_seq", params='{"seq_param_1": 1}')
seq.save()

# Make a basic task entry
subj = models.Subject.objects.get(name="test_subject")
task = models.Task.objects.get(name="tracking")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
te.save()

# Make a manual control task entry
task = models.Task.objects.get(name="manual control")
expm = models.Experimenter.objects.get(name="experimenter_1")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="task_desc",
                      session="test session", project="test project", params='{"task_param_1": 1}', sequence_id=seq.id)
te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 1}'
te.save()
te.feats.set([feat])
te.save()

# And a flash task entry
task = models.Task.objects.get(name="manual control")
expm = models.Experimenter.objects.get(name="experimenter_1")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="flash")
te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 0}'
te.save()

system = models.System(name="test_system", path="", archive="")
system.save()

# Add a bmi task entry
task = models.Task(name="bmi control")
task.save()
subj = models.Subject.objects.get(name="test_subject")
expm = models.Experimenter.objects.get(name="experimenter_1")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, params='{"bmi": 1}')
te.save()

decoder = models.Decoder(name="test_decoder", entry_id=te.id)
decoder.save()
