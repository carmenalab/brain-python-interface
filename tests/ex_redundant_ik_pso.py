#!/usr/bin/python
'''
Example of inverse kinematics using the simple gradient descent method
'''

from riglib.bmi import robot_arms
reload(robot_arms)
import numpy as np
import matplotlib.pyplot as plt
import time

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
print "target position"
print target_pos
q = q_sub[:]



def update_vel(v, q, pbest, gbest, w=0.25, c1=0.5, c2=0.25):
	phi1 = np.random.rand()
	phi2 = np.random.rand()
	return w*v + c1*phi1*(pbest - q) + c2*phi2*(gbest - q)


def cost(q, weight=10):
	return np.linalg.norm(q - q_start) + weight*np.linalg.norm(planar_chain.endpoint_pos(q) - target_pos)

import time
start_time = time.time()

# Initialize the particles; 
n_particles = 10
n_joints = planar_chain.n_joints
q_start = np.array([np.random.uniform(-pi, pi), np.random.uniform(0, pi), np.random.uniform(-pi/2, pi/2), np.random.uniform(-pi/2, 10*pi/180)])

from riglib.stereo_opengl import ik
noise = 5*np.random.randn(3)
noise[1] = 0
angles = ik.inv_kin_2D(target_pos + noise, 15., 25.)
q_start_constr = np.array([-angles[0][1], -angles[0][3], 0, 0])

n_iter = 10
particles_q = np.tile(q_start_constr, [n_particles, 1])
particles_v = np.random.randn(n_particles, n_joints)


gbest = particles_q.copy()
gbestcost = np.array(map(cost, gbest))
pbest = gbest[np.argmin(gbestcost)]
pbestcost = cost(pbest)

for k in range(n_iter):
	# update positions of particles
	particles_q += particles_v

	# apply joint limits
	particles_q = np.array(map(lambda x: planar_chain.apply_joint_limits(x)[0], particles_q))

	# update the costs
	costs = np.array(map(cost, particles_q))

	# update the 'bests'
	gbest[gbestcost > costs] = particles_q[gbestcost > costs]
	gbestcost = map(cost, gbest)

	pbest = gbest[np.argmin(gbestcost)]
	pbestcost = cost(pbest)	

	# update the velocity
	phi1 = np.random.rand()
	phi2 = np.random.rand()
	w=0.25
	c1=0.5
	c2=0.25
	particles_v = w*particles_v + c1*phi1*(np.tile(pbest, [n_particles, 1]) - particles_q) + c2*phi2*(gbest - particles_q)
	# for n in range(n_particles):
	# 	def update_vel(v, q, pbest, gbest, w=0.25, c1=0.5, c2=0.25):
	# 		return w*v + c1*phi1*(pbest - q) + c2*phi2*(gbest - q)
	# 	particles_v[n] = update_vel(particles_v[n], particles_q[n], pbest, gbest[n])

	if np.linalg.norm(planar_chain.endpoint_pos(pbest) - target_pos) < 0.5:
		break


end_time = time.time()
print planar_chain.endpoint_pos(pbest)
print np.linalg.norm(planar_chain.endpoint_pos(pbest) - target_pos)
print "pbest = "
print pbest * 180/np.pi
print "q_start_constr"
print q_start_constr * 180/np.pi
print "q_start"
print q_start * 180/np.pi
print "Runtime = %g" % (end_time-start_time)
