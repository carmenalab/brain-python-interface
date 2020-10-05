"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""
from django.test import TestCase, Client
import json, time, sys, datetime
import os
os.environ['DISPLAY'] = ':0'

from tracker import models
from tracker import tasktrack
from tracker import views
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

    #def test_data_file_linking_through_post(self):
    #    c = Client()
    #    post_data = {'file_path':"dummy_file_path", 'data_system_id':1}
    #    te = models.TaskEntry.objects.all()[0]
    #    c.post("/exp_log/link_data_files/%d/submit" % te.id, post_data)

    #    data_file = models.DataFile.objects.get(path="dummy_file_path")
    #    self.assertEqual(data_file.entry_id, te.id)


class TestModels(TestCase):
    def test_add_new_task_to_table(self):
        c = Client()

        post_data = {"name": "test_add_new_task_to_table",
            "import_path": "riglib.experiment.LogExperiment"}
        resp = c.post("/setup/add/new_task", post_data)

        task = models.Task.objects.get(name="test_add_new_task_to_table")
        task_cls = task.get()
        self.assertEqual(task_cls, LogExperiment)

    def test_add_new_feature_to_table(self):
        c = Client()
        post_data = {"name": "saveHDF",
            "import_path": "features.hdf_features.SaveHDF"}
        resp = c.post("/setup/add/new_feature", post_data)

        feat = models.Feature.objects.get(name="saveHDF")
        feat_cls = feat.get()
        from features.hdf_features import SaveHDF
        self.assertEqual(feat_cls, SaveHDF)

        feat.delete()
        self.assertEqual(len(models.Feature.objects.all()), 0)

    def test_add_new_subject_from_POST(self):
        test_name = "test_subject_post"
        c = Client()
        post_data = {"subject_name": test_name}

        resp = c.post("/setup/add/new_subject", post_data)

        subj = models.Subject.objects.get(name=test_name)
        self.assertEqual(subj.name, test_name)

    def test_add_built_in_feature_from_POST(self):
        c = Client()
        self.assertEqual(len(models.Feature.objects.all()), 0)

        post_data = {"saveHDF": 1}
        resp = c.post("/setup/add/enable_features", post_data)

        from features.hdf_features import SaveHDF
        feat = models.Feature.objects.get(name="saveHDF")
        self.assertEqual(feat.get(), SaveHDF)

        feat.delete()
        self.assertEqual(len(models.Feature.objects.all()), 0)

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

    def test_task_entry_collections(self):
        # setup
        subj = models.Subject(name="test_subject")
        subj.save()
        subj = models.Subject.objects.get(name="test_subject")

        task = models.Task(name="test_task")
        task.save()
        task = models.Task.objects.get(name="test_task")

        te1 = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te1.save()

        te2 = models.TaskEntry(subject_id=subj.id, task_id=task.id)
        te2.save()

        col = models.TaskEntryCollection(name="new_col")
        col.save()
        col.add_entry(te1)

        self.assertEqual(len(col.entries.all()), 1)

        # adding the same entry twice shouldn't do anything
        col.add_entry(te1)
        self.assertEqual(len(col.entries.all()), 1)

        # adding a second entry should increase the length of the list
        col.add_entry(te2)
        self.assertEqual(len(col.entries.all()), 2)

        # remove_entry should cause a change in the list
        col.remove_entry(te1)
        self.assertEqual(len(col.entries.all()), 1)
        self.assertEqual(col.entries.all()[0].id, te2.id)


class TestExpLog(TestCase):
    def test_list_exp_history(self):
        subj = models.Subject(name="test_subject")
        subj.save()
        subj = models.Subject.objects.get(name="test_subject")

        task = models.Task(name="test_task")
        task.save()
        task = models.Task.objects.get(name="test_task")

        list_data0 = views._list_exp_history()
        self.assertEqual(len(list_data0['entries']), 0)
        self.assertEqual(len(list_data0['subjects']), 1)
        self.assertEqual(len(list_data0['tasks']), 1)
        self.assertEqual(len(list_data0['features']), 0)
        self.assertEqual(len(list_data0['generators']), 0)

        te1 = models.TaskEntry(subject_id=subj.id, task_id=task.id, date=datetime.datetime.now())
        te1.save()

        list_data1 = views._list_exp_history()
        self.assertEqual(len(list_data1['entries']), 1)
        self.assertEqual(len(list_data1['subjects']), 1)
        self.assertEqual(len(list_data1['tasks']), 1)
        self.assertEqual(len(list_data1['features']), 0)
        self.assertEqual(len(list_data1['generators']), 0)


        for k in range(300):
            te2_date = datetime.datetime.now() - datetime.timedelta(days=k)
            te2 = models.TaskEntry(subject_id=subj.id, task_id=task.id, date=te2_date)
            te2.save()

        # all entries returned if no args
        list_data2 = views._list_exp_history()
        self.assertEqual(len(list_data2['entries']), 301)

        # 'listall' should return all entries
        list_data3 = views._list_exp_history(max_entries=100)
        self.assertEqual(len(list_data3['entries']), 100)

class TestGenerators(TestCase):
    def test_generator_retreival(self):
        task = models.Task(name="test_task1", import_path="riglib.experiment.mocks.MockSequenceWithGenerators")
        task.save()

        models.Generator.populate()
        self.assertEqual(len(models.Generator.objects.all()), 2)


class TestVisualFeedbackTask(TestCase):
    def test_start_experiment_python(self):
        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="test_vfb", import_path="built_in_tasks.passivetasks.TargetCaptureVFB2DWindow")
        task.save()

        models.Generator.populate()
        gen = models.Generator.objects.get(name='centerout_2D_discrete')

        import json
        seq_params = dict(nblocks=1, ntargets=1)
        seq_rec = models.Sequence(generator=gen,
            params=json.dumps(seq_params), task=task)
        seq_rec.save()
        print(seq_rec)

        task_rec = models.Task.objects.get(name='test_vfb')
        te = models.TaskEntry(task=task_rec, subject=subj)
        te.save()

        from built_in_tasks.passivetasks import TargetCaptureVFB2DWindow

        seq, seq_params = seq_rec.get()
        # seq = TargetCaptureVFB2DWindow.centerout_2D_discrete()

        base_class = task.get_base_class()
        from riglib import experiment
        from tracker import json_param
        Task = experiment.make(base_class, feats=[])

        params = json_param.Parameters.from_dict(dict(window_size=(480, 240)))
        params.trait_norm(Task.class_traits())


        from features import Autostart

        saveid = te.id
        task_start_data = dict(subj=subj.id, base_class=base_class, feats=[Autostart],
                      params=dict(window_size=(480, 240)), seq=seq_rec, seq_params=seq_params,
                      saveid=saveid)

        try:
            import pygame
            tracker = tasktrack.Track.get_instance()
            tracker.runtask(**task_start_data)
        except:
            print("Skipping test due to pygame missing")


class TestTaskStartStop(TestCase):
    def test_start_experiment_python(self):
        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp", import_path="riglib.experiment.LogExperiment")
        task.save()

        task_start_data = dict(subj=subj.id, base_class=task.get_base_class(), feats=[],
                      params=dict())

        # task_start_data = dict(subj=1, task=1, feats=dict(), params=dict(), sequence=None)
        tracker = tasktrack.Track.get_instance()
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

        tracker = tasktrack.Track.get_instance()
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

    def test_start_experiment_with_features(self):
        c = Client()

        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp", import_path="riglib.experiment.LogExperiment")
        task.save()

        feat = models.Feature(name="saveHDF", import_path="features.hdf_features.SaveHDF")
        feat.save()

        task_start_data = dict(subject=1, task=1, feats={"saveHDF":"saveHDF"}, params=dict(), sequence=None)

        post_data = {"data": json.dumps(task_start_data)}

        # if sys.platform == "win32":
        start_resp = c.post("/test", post_data)
        start_resp_obj = json.loads(start_resp.content.decode("utf-8"))

        tracker = tasktrack.Track.get_instance()
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

class TestTaskAnnotation(TestCase):
    def test_annotate_experiment(self):
        c = Client()

        subj = models.Subject(name="test_subject")
        subj.save()

        task = models.Task(name="generic_exp", import_path="riglib.experiment.mocks.MockSequenceWithGenerators")
        task.save()

        models.Generator.populate()

        feat = models.Feature(name="saveHDF", import_path="features.hdf_features.SaveHDF")
        feat.save()

        task_start_data = dict(subject=1, task=1, feats={"saveHDF":"saveHDF"}, params=dict(),
            sequence=dict(generator=1, name="seq1", params=dict(n_targets=1000), static=False))

        post_data = {"data": json.dumps(task_start_data)}

        # if sys.platform == "win32":
        start_resp = c.post("/test", post_data)
        start_resp_obj = json.loads(start_resp.content.decode("utf-8"))

        tracker = tasktrack.Track.get_instance()
        h5file = tracker.task_proxy.get_h5_filename()
        self.assertTrue(tracker.task_running())

        time.sleep(2)
        c.post("/exp_log/record_annotation", dict(annotation="test post annotation"))

        time.sleep(2)
        stop_resp = c.post("/exp_log/stop/")

        time.sleep(2)
        self.assertFalse(tracker.task_running())

        # check that the annotation is recorded in the HDF5 file
        import h5py
        hdf = h5py.File(h5file)
        self.assertTrue(b"annotation: test post annotation" in hdf["/task_msgs"]["msg"][:])


class TestParamCast(TestCase):
    def test_norm_trait(self):
        from tracker import json_param
        from riglib.experiment import traits

        t = traits.Float(1, descr='test trait')
        t1 = json_param.norm_trait(t, 1.0)
        self.assertEqual(t1, 1.0)

        #self.assertRaises(Exception, json_param.norm_trait, t, '1.0')
