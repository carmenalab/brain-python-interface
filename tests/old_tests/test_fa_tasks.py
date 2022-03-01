from tasks import factor_analysis_tasks
from riglib import experiment
import pickle
from ..features.plexon_features import PlexonBMI
from ..features.hdf_features import SaveHDF
from riglib.bmi import train

# decoder = pickle.load(open('/Users/preeyakhanna/Dropbox/Carmena_Lab/FA_exp/grom_data/grom20151201_01_RMLC12011916.pkl'))

# Task = experiment.make(factor_analysis_tasks.FactorBMIBase, [CorticalBMI, SaveHDF])
# targets = factor_analysis_tasks.FactorBMIBase.generate_catch_trials()
# kwargs=dict(session_length=20.)
# task = Task(targets, plant_type="cursor_25x14", **kwargs)
# task.decoder = decoder

# import riglib.plexon
# task.sys_module = riglib.plexon

# task.init()
# task.run()


from tasks import choice_fa_tasks
decoder = pickle.load(open('/storage/decoders/grom20160201_01_RMLC02011515_w_fa_dict_from_4048.pkl'))

Task = experiment.make(choice_fa_tasks.FreeChoiceFA, [PlexonBMI, SaveHDF])
targets = choice_fa_tasks.FreeChoiceFA.centerout_2D_discrete_w_free_choice()
task = Task(targets, plant_type="cursor_25x14")
task.decoder =decoder
