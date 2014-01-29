#!/usr/bin/python
'''
Setup for visual feedback task in joint space. This should try to compensate for
slow IK
'''
import numpy as np
from tasks import performance
import matplotlib.pyplot as plt
from riglib.bmi import robot_arms, assist, train, goal_calculators
from riglib.stereo_opengl import ik
import time

from riglib import mp_calc
reload(mp_calc)

reload(assist)
reload(robot_arms)
reload(goal_calculators)

pi = np.pi

te = performance._get_te(3086)
target = te.hdf.root.task[:]['target']

starting_pos_ws = np.array([5., 0, 5])
shoulder_anchor = np.array([2, 0, -15])

# Generate the list of targets
from tasks import generatorfunctions
target_list = generatorfunctions.centerout_2D_discrete(nblocks=1, ntargets=2)
target_list = target_list.reshape(-1, 3)

n_targets = target_list.shape[0]

# Initialize the kinematic chain
chain = robot_arms.PlanarXZKinematicChain([15, 15, 5, 5])
chain.joint_limits = [(-pi, pi), (-pi, 0), (-pi/2, pi/2), (-pi/2, 10*pi/180)]


### Initialize the state of the arm
starting_pos_ps = starting_pos_ws - shoulder_anchor
q_start = chain.random_sample()

noise = 5*np.random.randn(3)
noise[1] = 0
angles = ik.inv_kin_2D(starting_pos_ps + noise, 15., 25.)
q_start_constr = np.array([-angles[0][1], -angles[0][3], 0, 0])

joint_pos = chain.inverse_kinematics_pso(starting_pos_ps, q_start_constr, verbose=True, time_limit=1.0, n_particles=30)
cursor_pos = chain.endpoint_pos(joint_pos)
print "starting position"
print starting_pos_ps
print "error = %g" % np.linalg.norm(cursor_pos - starting_pos_ps)

assister = assist.TentacleAssist(ssm=train.tentacle_2D_state_space, kin_chain=chain)

q_start = chain.inverse_kinematics(np.array([5., 0., 5.]) - shoulder_anchor, verbose=True, n_particles=500, eps=0.05, n_iter=10)
x_init = np.hstack([q_start, np.zeros_like(q_start), 1])
x_init = np.mat(x_init).reshape(-1, 1)
x = [x_init]
goal_calc = goal_calculators.PlanarMultiLinkJointGoal(train.tentacle_2D_state_space, shoulder_anchor, chain, multiproc=True, init_resp=x_init)

target_idx = 0
k = 0
cursor_pos_hist = []

target_joint_pos = chain.inverse_kinematics(target_list[target_idx] - shoulder_anchor, verbose=True, n_particles=500, eps=0.05, n_iter=10)
target = target_list[target_idx] - shoulder_anchor	
while True:
	if k % 100 == 0: print k
	joint_pos = np.array(x[-1][0:4, -1]).ravel()

	# evaluate the cursor position
	cursor_pos = chain.endpoint_pos(joint_pos)
	cursor_pos_hist.append(cursor_pos)	
	if np.linalg.norm(cursor_pos - target) < 2.:
		print target_idx
		target_idx += 1
		if target_idx >= n_targets: break
		target = target_list[target_idx] - shoulder_anchor	

	target_state = goal_calc(target_list[target_idx], verbose=True, n_particles=500, eps=0.05, n_iter=10)
	target_state = np.mat(target_state.reshape(-1, 1))

	current_state = x[-1]
	x_next = assister(current_state, target_state)

	x.append(x_next)
	k += 1


cursor_pos_hist = np.vstack(cursor_pos_hist)
endpoint_locations = target_list - shoulder_anchor

plt.figure()
plt.plot(cursor_pos_hist[:,0], cursor_pos_hist[:,2])
plt.scatter(endpoint_locations[:,0], endpoint_locations[:,2])
plt.show()