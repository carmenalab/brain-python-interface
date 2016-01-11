from tasks import factor_analysis_tasks
from riglib import experiment
import pickle
from features.neural_sys_features import CorticalBMI
from features.hdf_features import SaveHDF

decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/FA_exp/grom_data/grom20151201_01_RMLC12011916.pkl'))

Task = experiment.make(factor_analysis_tasks.FactorBMIBase, [CorticalBMI, SaveHDF])
targets = factor_analysis_tasks.FactorBMIBase.generate_catch_trials()
kwargs=dict(session_length=20.)
task = Task(targets, plant_type="cursor_25x14", **kwargs)
task.decoder = decoder

import riglib.plexon
task.sys_module = riglib.plexon

task.init()
task.run()