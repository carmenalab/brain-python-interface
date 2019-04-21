"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""
from django.test import TestCase, Client
import json

from tracker import models
from tracker import exp_tracker
import psutil

from riglib.experiment import LogExperiment

class TestModels(TestCase):
    def test_add_new_task_to_table(self):
        c = Client()

        post_data = {"name": "test_add_new_task_to_table", 
            "import_path": "riglib.experiment.LogExperiment"}
        resp = c.post("/setup/add_new_task", post_data)

        task = models.Task.objects.get(name="test_add_new_task_to_table")
        task_cls = task.get()
        self.assertEqual(task_cls, LogExperiment)

    # def test_add_new_task_no_features(self):
    #     task = models.Task(name="test_task", import_path="riglib.experiment.LogExperiment")
    #     task.save()
    #     task = models.Task.objects.get(name="test_task")

    #     task_cls = task.get()
    #     from riglib.experiment import LogExperiment
    #     self.assertEqual(task_cls, LogExperiment)

    # def test_create_task_entry(self):
    #     """
    #     Tests that 1 + 1 always equals 2.
    #     """
    #     subj = models.Subject(name="test_subject")
    #     subj.save()
    #     subj = models.Subject.objects.get(name="test_subject")

    #     task = models.Task(name="test_task")
    #     task.save()
    #     task = models.Task.objects.get(name="test_task")

    #     te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
    #     te.save()
    #     from tracker import tasktrack

# class TestTaskStartStop(TestCase):
#     def test_start_experiment(self):
#         c = Client()

#         subj = models.Subject(name="test_subject")
#         subj.save()

#         task = models.Task(name="generic_exp")
#         task.save()        

#         task_start_data = dict(subject=1, task=1, feats=dict(), params=dict(), sequence=None)

#         post_data = {"data": json.dumps(task_start_data)}
#         start_resp = c.post("/start", post_data)
#         start_resp_obj = json.loads(start_resp.content.decode("utf-8"))

#         import time
#         time.sleep(2)
        
#         stop_resp = c.post("/exp_log/stop/")
#         print("stop_resp", stop_resp)
#         print(stop_resp.content)

#     def tearDown(self):
#         p = psutil.Process(exp_tracker.get().websock.pid)
#         p.terminate()

class TestWebsocket(TestCase):
    def test_send(self):
        pass
        # serv = Server()
        # serv.send(dict(state="working well"))
        # serv.stop()
