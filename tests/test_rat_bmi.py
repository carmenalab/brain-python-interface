
from features.hdf_features import SaveHDF
from features.plexon_features import PlexonBMI
from features.generator_features import Autostart
from features.arduino_features import PlexonSerialDIORowByte
from features.reward_features import RewardSystem
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, State, offset_state

from riglib import experiment
import plantlist

import numpy as np
import os, shutil, pickle
import pickle
import time, datetime

from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi import feedback_controllers

from tasks import rat_bmi_tasks
from db.tracker import models
from db.json_param import Parameters

decoder_trained_id = 5402
decoder_list_ix = 1
session_length = 10*60.

decoder_list = models.Decoder.objects.filter(entry=decoder_trained_id)
Decoder = decoder_list[decoder_list_ix]
decoder = pickle.load(open('/storage/decoders/'+Decoder.path))

kw=dict(decoder=decoder)
s = models.Subject.objects.filter(name='Gromit')
t = models.Task.objects.filter(name='rat_bmi')
entry = models.TaskEntry(subject_id=s[0].id, task=t[0])
entry.sequence_id = -1

params = Parameters.from_dict(dict(decoder=Decoder.pk, decoder_path=Decoder.path))
entry.params = params.to_json()
entry.save()

saveid = entry.id

Task = experiment.make(rat_bmi_tasks.RatBMI, [Autostart, PlexonBMI, PlexonSerialDIORowByte, RewardSystem, SaveHDF])
Task.pre_init(saveid=saveid)
task = Task(plant_type='aud_cursor', session_length=session_length., **kw)
task.subj=s[0]

task.init()
task.run()

#Cleanup
from db.tracker import dbq
cleanup_successful = task.cleanup(dbq, saveid, subject=task.subj)
task.decoder.save()
task.cleanup_hdf()
task.terminate()