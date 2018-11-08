"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""
from django.test import TestCase, Client
import json

from tracker import models

class TestModels(TestCase):
    def test_create_task_entry(self):
        """
        Tests that 1 + 1 always equals 2.
        """
        subj = models.Subject(name="test_subject")
        subj.save()
        subj = models.Subject.objects.get(name="test_subject")

        task = models.Task(name="test_task")
        task.save()
        task = models.Task.objects.get(name="test_task")

        te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te.save()
        from tracker import tasktrack

class TestTaskStartStop(TestCase):
    def test_start_experiment(self):
        c = Client()

        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp")
        task.save()        

        task_start_data = dict(subject=1, task=1, feats=dict(), params=dict(), sequence=None)

        post_data = {"data": json.dumps(task_start_data)}
        start_resp = c.post("/start", post_data)
        start_resp_obj = json.loads(start_resp.content.decode("utf-8"))

        import time
        time.sleep(2)
        
        stop_resp = c.post("/exp_log/stop/")
        print("stop_resp", stop_resp)
        print(stop_resp.content)