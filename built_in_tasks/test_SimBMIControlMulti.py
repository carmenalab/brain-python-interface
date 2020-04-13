from bmimultitasks import SimBMIControlMulti
import numpy as np
#build a sequence generator
N_TARGETS = 8
N_TRIALS = 6
seq = SimBMIControlMulti.sim_target_seq_generator_multi(N_TARGETS, N_TRIALS)

#build a observer matrix
N_NEURONS  = 20
N_STATES = 7 #3 positions and 3 velocities and an offset

sim_C = np.zeros((N_NEURONS,N_STATES))
#control x positive directions
sim_C[2,:] = np.array([0,0,0,1,0,0,0])
sim_C[3,:] = np.array([0,0,0,-1,0,0,0])
#control z positive directions
sim_C[5,:] = np.array([0,0,0,0,0,1,0])
sim_C[6,:] = np.array([0,0,0,0,0,-1,0])

#set up assist level
assist_level = (0.1, 0.1)

exp = SimBMIControlMulti(seq,sim_C = sim_C, assist_level = assist_level)
exp.init()
exp.run()