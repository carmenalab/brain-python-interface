from bmimultitasks import SimBMIControlMulti
from ..features import SaveHDF
import numpy as np
from riglib import experiment

# build a sequence generator
if __name__ == "__main__":
    N_TARGETS = 8
    N_TRIALS = 6
    seq = SimBMIControlMulti.sim_target_seq_generator_multi(
        N_TARGETS, N_TRIALS)

    # build a observer matrix
    N_NEURONS = 20
    N_STATES = 7  # 3 positions and 3 velocities and an offset

    # build the observation matrix
    sim_C = np.zeros((N_NEURONS, N_STATES))
    # control x positive directions
    sim_C[2, :] = np.array([0, 0, 0, 1, 0, 0, 0])
    sim_C[3, :] = np.array([0, 0, 0, -1, 0, 0, 0])
    # control z positive directions
    sim_C[5, :] = np.array([0, 0, 0, 0, 0, 1, 0])
    sim_C[6, :] = np.array([0, 0, 0, 0, 0, -1, 0])

    # set up assist level
    assist_level = (0.1, 0.1)

    #exp = SimBMIControlMulti(seq,sim_C = sim_C, assist_level = assist_level)
    # exp.init()
    # exp.run()

    kwargs = dict()

    kwargs['sim_C'] = sim_C
    kwargs['assist_level'] = assist_level

    base_class = SimBMIControlMulti
    feats = [SaveHDF]
    #feats = []
    Exp = experiment.make(base_class, feats=feats)
    print(Exp)


    exp = Exp(seq, **kwargs)
    exp.init()
    exp.run()  # start the task
