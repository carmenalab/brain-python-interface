#!/usr/bin/python
'''
Example of inverse kinematics using the simple gradient descent method
'''

from riglib.bmi import robot_arms
reload(robot_arms)
import numpy as np
import matplotlib.pyplot as plt
import time
from riglib.stereo_opengl import ik
import cProfile

pi = np.pi

q = np.array([0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * pi/180
q_sub = q[1::3]

chain = robot_arms.KinematicChain([15, 15, 5, 5])
[t, allt] = chain.forward_kinematics(q);

planar_chain = robot_arms.PlanarXZKinematicChain([15, 15, 5, 5])
[t, allt] = planar_chain.forward_kinematics(q_sub);

# TODO check the sign for the finger joint limits
inf = np.inf
planar_chain.joint_limits = [(-pi, pi), (-pi, 0), (-pi/2, pi/2), (-pi/2, 10*pi/180)]


# target_pos = np.array([10, 0, 10])
shoulder_anchor = np.array([2, 0, -15])
x_target_pos = (np.random.randn() - 0.5)*25
z_target_pos = (np.random.randn() - 0.5)*14
target_pos = np.array([x_target_pos, 0, z_target_pos]) - shoulder_anchor
target_pos = np.array([-14.37130744,   0.        ,  22.97472612])

q = q_sub[:]

def cost(q, q_start, weight=10):
	return np.linalg.norm(q - q_start) + weight*np.linalg.norm(planar_chain.endpoint_pos(q) - target_pos)

def stuff():
	# Initialize the particles; 
	n_particles = 10
	n_joints = planar_chain.n_joints
	q_start = np.array([np.random.uniform(-pi, pi), np.random.uniform(0, pi), np.random.uniform(-pi/2, pi/2), np.random.uniform(-pi/2, 10*pi/180)])

	noise = 5*np.random.randn(3)
	noise[1] = 0
	angles = ik.inv_kin_2D(target_pos + noise, 15., 25.)
	q_start_constr = np.array([-angles[0][1], -angles[0][3], 0, 0])

	n_iter = 10
	particles_q = np.tile(q_start_constr, [n_particles, 1])
	particles_v = np.random.randn(n_particles, n_joints)

	cost_fn = lambda x: cost(x, q_start)

	gbest = particles_q.copy()
	gbestcost = np.array(map(cost_fn, gbest))
	pbest = gbest[np.argmin(gbestcost)]
	pbestcost = cost_fn(pbest)

	min_limits = np.array([x[0] for x in planar_chain.joint_limits])
	max_limits = np.array([x[1] for x in planar_chain.joint_limits])
	min_limits = np.tile(min_limits, [n_particles, 1])
	max_limits = np.tile(max_limits, [n_particles, 1])

	start_time = time.time()
	for k in range(n_iter):
		# update positions of particles
		particles_q += particles_v

		# apply joint limits
		# particles_q = np.array(map(lambda x: planar_chain.apply_joint_limits(x)[0], particles_q))
		min_viol = particles_q < min_limits
		max_viol = particles_q > max_limits
		particles_q[min_viol] = min_limits[min_viol]
		particles_q[max_viol] = max_limits[max_viol]

		# update the costs
		costs = np.array(map(cost_fn, particles_q))

		# update the 'bests'
		gbest[gbestcost > costs] = particles_q[gbestcost > costs]
		gbestcost = map(cost_fn, gbest)

		pbest = gbest[np.argmin(gbestcost)]
		pbestcost = cost_fn(pbest)	

		# update the velocity
		phi1 = 1#np.random.rand()
		phi2 = 1#np.random.rand()
		w=0.25
		c1=0.5
		c2=0.25
		particles_v = w*particles_v + c1*phi1*(np.tile(pbest, [n_particles, 1]) - particles_q) + c2*phi2*(gbest - particles_q)

		if np.linalg.norm(planar_chain.endpoint_pos(pbest) - target_pos) < 0.5:
			break
		
	end_time = time.time()
	print "Runtime = %g" % (end_time-start_time)

	return pbest

starting_pos = np.array([-5., 0, 5])
target_pos = starting_pos - shoulder_anchor

q_start = planar_chain.random_sample()

noise = 5*np.random.randn(3)
noise[1] = 0
angles = ik.inv_kin_2D(target_pos + noise, 15., 25.)
q_start_constr = np.array([-angles[0][1], -angles[0][3], 0, 0])


pbest = planar_chain.inverse_kinematics_pso(q_start_constr, target_pos, verbose=True, time_limit=0.010)

# cProfile.run('planar_chain.inverse_kinematics_pso(q_start_constr, target_pos)')


# print planar_chain.endpoint_pos(pbest)
print "target position"
print target_pos
print "error = %g" % np.linalg.norm(planar_chain.endpoint_pos(pbest) - target_pos)

# print "q_start_constr"
# print q_start_constr * 180/np.pi
# print "q_start"
# print q_start * 180/np.pi

