import unittest
import tasks
from analysis import performance
from tasks import generatorfunctions as genfns
import numpy as np 

from tasks import bmi_recon_tasks
from tasks.bmimultitasks import SimBMIControlMulti
reload(bmi_recon_tasks)
reload(tasks)

from tasks.bmi_recon_tasks import BMIReconstruction

from riglib.bmi import goal_calculators
class TentacleAttractorBMIRecon(BMIReconstruction):
    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine what the target state of the task is
        '''
        target_loc = np.zeros(3) 
        ## The line above is the only change between this task and the BMIControlMulti task

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoalCached):
            task_eps = np.inf
        else:
            task_eps = 0.5
        ik_eps = task_eps/10
        data, solution_updated = self.goal_calculator(target_loc, verbose=False, n_particles=500, eps=ik_eps, n_iter=10, q_start=-self.plant.get_intrinsic_coordinates())
        target_state, error = data

        if isinstance(self.goal_calculator, goal_calculators.PlanarMultiLinkJointGoal) and error > task_eps and solution_updated:
            self.goal_calculator.reset()

        return np.tile(np.array(target_state).reshape(-1,1), [1, self.decoder.n_subbins])


idx = 2284

te = performance.BMIControlMultiTaskEntry(idx, dbname='testing')
n_iter = len(te.hdf.root.task)
        
gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)

cls = TentacleAttractorBMIRecon
self = task = cls(te, n_iter, gen)
from tasks import plantlist
task.plant = plantlist.chain_15_15_5_5
task.init()
        
error = task.calc_recon_error(verbose=False)
abs_max_error = np.max(np.abs(error))

print abs_max_error
