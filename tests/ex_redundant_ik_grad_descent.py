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

q = np.array([0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 20, 0]) * pi/180
q_sub = q[1::3]

chain = robot_arms.KinematicChain([15, 15, 5, 5])
[t, allt] = chain.forward_kinematics(q);

planar_chain = robot_arms.PlanarXZKinematicChain([15, 15, 5, 5])
[t, allt] = planar_chain.forward_kinematics(q_sub);

# TODO check the sign for the finger joint limits
inf = np.inf
planar_chain.joint_limits = [(-inf, inf), (-inf, inf), (-pi/2, pi/2), (-pi/2, 10*pi/180)]

target_pos = np.array([10, 0, 10])
q = q_sub[:]

q_star, path = planar_chain.inverse_kinematics(q_sub.copy(), target_pos, verbose=True, return_path=True)

# plt.close('all')
# planar_chain.plot(q_star)
print planar_chain.endpoint_pos(q_star)

# plt.figure()
# plt.plot(endpoint_traj[:k,0], endpoint_traj[:k,2])	
# plt.show()