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

# Make a basic task entry
subj = models.Subject.objects.get(name="test_subject")
task = models.Task.objects.get(name="tracking")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
te.save()

# Make a manual control task entry
task = models.Task.objects.get(name="manual control")
expm = models.Experimenter.objects.get(name="experimenter_1")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="task_desc")
te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 1}'
te.save()

# And a flash task entry
task = models.Task.objects.get(name="manual control")
expm = models.Experimenter.objects.get(name="experimenter_1")
te = models.TaskEntry(subject_id=subj.id, task_id=task.id, experimenter_id=expm.id, entry_name="flash")
te.report = '{"runtime": 3.0, "n_trials": 2, "n_success_trials": 0}'
te.save()

system = models.System(name="test_system", path="", archive="")
system.save()