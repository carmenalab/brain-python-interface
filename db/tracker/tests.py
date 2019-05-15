"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""
from django.test import TestCase, Client
import json
import time
import sys

from db.tracker import models
from db.tracker import exp_tracker
# import psutil

from riglib.experiment import LogExperiment


class TestDataFile(TestCase):
    def setUp(self):
        subj = models.Subject(name="test_subject")
        subj.save()
        subj = models.Subject.objects.get(name="test_subject")

        task = models.Task(name="test_task")
        task.save()
        task = models.Task.objects.get(name="test_task")

        te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te.save()

        system = models.System(name="test_system", path="", archive="")
        system.save()

    def test_file_linking(self):
        te = models.TaskEntry.objects.all()[0]
        system = models.System.objects.get(name="test_system")
        data_file = models.DataFile(path="dummy_file_path", system_id=system.id, entry_id=te.id)
        data_file.save()

        data_file = models.DataFile.objects.get(path="dummy_file_path")

        self.assertTrue(data_file.entry_id == te.id)
        self.assertTrue(data_file.system_id == system.id)

    def test_data_file_linking_through_post(self):
        c = Client()
        post_data = {'file_path':"dummy_file_path", 'data_system_id':1}
        te = models.TaskEntry.objects.all()[0]
        c.post("/exp_log/link_data_files/%d/submit" % te.id, post_data)

        data_file = models.DataFile.objects.get(path="dummy_file_path")
        self.assertEqual(data_file.entry_id, te.id)


class TestModels(TestCase):
    def test_add_new_task_to_table(self):
        c = Client()

        post_data = {
            "name": "test_add_new_task_to_table",
            "import_path": "riglib.experiment.LogExperiment"
        }
        resp = c.post("/setup/add/new_task", post_data)

        task = models.Task.objects.get(name="test_add_new_task_to_table")
        task_cls = task.get()
        self.assertEqual(task_cls, LogExperiment)

    def test_add_new_task_no_features(self):
        task = models.Task(name="test_task", import_path="riglib.experiment.LogExperiment")
        task.save()
        task = models.Task.objects.get(name="test_task")

        task_cls = task.get()
        from riglib.experiment import LogExperiment
        self.assertEqual(task_cls, LogExperiment)

    def test_create_task_entry(self):
        """
        """
        subj = models.Subject(name="test_subject")
        subj.save()
        subj = models.Subject.objects.get(name="test_subject")

        task = models.Task(name="test_task")
        task.save()
        task = models.Task.objects.get(name="test_task")

        te = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te.save()


class TestTaskStartStop(TestCase):
    def test_start_experiment_python(self):
        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp", import_path="riglib.experiment.LogExperiment")
        task.save()        

        task_start_data = dict(subj=subj.id, base_class=task.get_base_class(), feats=[],
                      params=dict())

        # task_start_data = dict(subj=1, task=1, feats=dict(), params=dict(), sequence=None)
        tracker = exp_tracker.get()
        tracker.runtask(**task_start_data)

        time.sleep(5)
        tracker.stoptask()

    def test_start_experiment_ajax(self):
        c = Client()

        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp", import_path="riglib.experiment.LogExperiment")
        task.save()        

        task_start_data = dict(subject=1, task=1, feats=dict(), params=dict(), sequence=None)

        post_data = {"data": json.dumps(task_start_data)}

        # if sys.platform == "win32":
        start_resp = c.post("/test", post_data)
        start_resp_obj = json.loads(start_resp.content.decode("utf-8"))
        print("JSON response")
        print(start_resp_obj)

        tracker = exp_tracker.get()
        self.assertTrue(tracker.task_running())

        # check the 'state' of the task
        self.assertEqual(tracker.task_proxy.get_state(), "wait")

        # update report stats 
        tracker.task_proxy.update_report_stats()

        # access report stats
        reportstats = tracker.task_proxy.reportstats
        self.assertTrue(len(reportstats.keys()) > 0)

        time.sleep(2)
        stop_resp = c.post("/exp_log/stop/")

        time.sleep(2)
        self.assertFalse(tracker.task_running())


class TestWebsocket(TestCase):
    def test_send(self):
        pass
        # serv = Server()
        # serv.send(dict(state="working well"))
        # serv.stop()
