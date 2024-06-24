import unittest
import numpy as np 
import plantlist
from tasks import bmi_recon_tasks
import dbfunctions as dbfn

from riglib.bmi import goal_calculators

class TentacleAttractorBMIRecon(bmi_recon_tasks.BMIReconstruction):
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

    def create_assister(self):
        from tasks.bmimultitasks import TentacleAssist
        self.assister = TentacleAssist(ssm=self.decoder.ssm, kin_chain=self.plant.kin_chain, update_rate=self.decoder.binlen)
        print(self.assister)

    def create_goal_calculator(self):
        shoulder_anchor = self.plant.base_loc
        chain = self.plant.kin_chain
        q_start = self.plant.get_intrinsic_coordinates()
        x_init = np.hstack([q_start, np.zeros_like(q_start), 1])
        x_init = np.mat(x_init).reshape(-1, 1)

        cached = True

        if cached:
            goal_calc_class = goal_calculators.PlanarMultiLinkJointGoalCached
            multiproc = False
        else:
            goal_calc_class = goal_calculators.PlanarMultiLinkJointGoal
            multiproc = True

        self.goal_calculator = goal_calc_class(self.decoder.ssm, shoulder_anchor, 
                                               chain, multiproc=multiproc, init_resp=x_init)



idx = 2284
te = dbfn.TaskEntry(idx, dbname='testing')
n_iter = len(te.hdf.root.task)
        

cls = TentacleAttractorBMIRecon
gen = []
task = cls(te, n_iter)

task.plant = plantlist.chain_15_15_5_5
task.init()
        
error = task.calc_recon_error(verbose=False, n_iter_betw_fb=1000)
abs_max_error = np.max(np.abs(error))

print(abs_max_error)
