from tasks import factor_analysis_tasks
from riglib import experiment
import pickle

decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/FA_exp/grom_data/grom20151201_01_RMLC12011916.pkl'))

Task = experiment.make(factor_analysis_tasks.FactorBMIBase)
targets = factor_analysis_tasks.FactorBMIBase.generate_catch_trials()
kwargs=dict(session_length=5.)
task = Task(targets, plant_type="cursor_25x14", **kwargs)
task.decoder = decoder
task.init()
task.run()