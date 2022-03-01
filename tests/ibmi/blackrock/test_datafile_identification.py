from ismore.invasive import bmi_ismoretasks
from riglib import experiment
from ..features.hdf_features import SaveHDF
from ..features.arduino_features import BlackrockSerialDIORowByte
import numpy as np

targets = bmi_ismoretasks.SimBMIControl.rehand_simple(length=100)
Task = experiment.make(bmi_ismoretasks.VisualFeedback, [SaveHDF, BlackrockSerialDIORowByte])
kwargs=dict(session_length=15., assist_level = (1., 1.), assist_level_time=60.,
    timeout_time=60.,)
task = Task(targets, plant_type='ReHand', **kwargs)
task.init()
