import unittest

from riglib.experiment import LogExperiment, FSMTable, StateTransitions, Sequence

event1to2 = [False, True,  False, False, False, False, False, False]
event1to3 = [False, False, False, False, True,  False, False, False]
event2to3 = [False, False, True,  False, False, False, False, False]
event2to1 = [False, False, False, False, False, False, False, False]
event3to2 = [False, False, False, False, False, True,  False, False]
event3to1 = [False, False, False, True,  False, False, False, False]

class MockLogExperiment(LogExperiment):
    status = FSMTable(
        state1=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='state1'),
        state3=StateTransitions(event3to2='state2', event3to1='state1'),
    )
    state = 'state1'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        super(MockLogExperiment, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockLogExperiment, self)._cycle()

    def _start_state3(self): pass
    def _while_state3(self): pass
    def _end_state3(self): pass
    def _start_state2(self): pass
    def _while_state2(self): pass
    def _end_state2(self): pass
    def _start_state1(self): pass
    def _while_state1(self): pass
    def _end_state1(self): pass
    ################## State trnasition test functions ##################
    def _test_event3to1(self, time_in_state): return event3to1[self.iter_idx]
    def _test_event3to2(self, time_in_state): return event3to2[self.iter_idx]
    def _test_event2to3(self, time_in_state): return event2to3[self.iter_idx]
    def _test_event2to1(self, time_in_state): return event2to1[self.iter_idx]
    def _test_event1to3(self, time_in_state): return event1to3[self.iter_idx]
    def _test_event1to2(self, time_in_state): return event1to2[self.iter_idx]
    def _test_stop(self, time_in_state):
        return self.iter_idx >= len(event1to2) - 1

    def get_time(self):
        return self.iter_idx

class MockDatabase(object):
    """ mock for dbq module """
    def save_log(self, idx, log, dbname='default'):
        f = open(str(idx), "w")
        f.write(str(log))
        f.close()

class TestLogExperiment(unittest.TestCase):
    def setUp(self):
        self.exp = MockLogExperiment()

    def test_exp_fsm_output(self):
        self.exp.run_sync()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), 
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

    def test_logging(self):
        mock_db = MockDatabase()
        self.exp.cleanup(mock_db, "id_is_test_file")
        self.assertEqual(str(self.exp.event_log), open("id_is_test_file").readlines()[0].rstrip())
        if os.path.exists("id_is_test_file"):
            os.remove("id_is_test_file")


class MockSequence(Sequence):
    event1to2 = [False, True,  False, True, False, True, False, True, False, True, False]
    event1to3 = [False, False, False, False, False,  False, False, False, False, False, False]
    event2to3 = [False, False, False,  False, False, False, False, False, False, False, False]
    event2to1 = [False, False, True, False, True, False, True, False, True, False, False]
    event3to2 = [False, False, False, False, False, False,  False, False, False, False, False]
    event3to1 = [False, False, False, False,  False, False, False, False, False, False, False]


    status = FSMTable(
        wait=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='wait'),
        state3=StateTransitions(event3to2='state2', event3to1='wait'),
    )
    state = 'wait'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        self.target_history = []
        super(MockSequence, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockSequence, self)._cycle()

    def _start_state3(self): pass
    def _while_state3(self): pass
    def _end_state3(self): pass
    def _start_state2(self): pass
    def _while_state2(self): pass
    def _end_state2(self): pass
    def _start_state1(self): pass
    def _while_state1(self): pass
    def _end_state1(self): pass
    ################## State trnasition test functions ##################
    def _test_event3to1(self, time_in_state): return self.event3to1[self.iter_idx]
    def _test_event3to2(self, time_in_state): return self.event3to2[self.iter_idx]
    def _test_event2to3(self, time_in_state): return self.event2to3[self.iter_idx]
    def _test_event2to1(self, time_in_state): return self.event2to1[self.iter_idx]
    def _test_event1to3(self, time_in_state): return self.event1to3[self.iter_idx]
    def _test_event1to2(self, time_in_state): return self.event1to2[self.iter_idx]
    def _test_stop(self, time_in_state):
        return self.iter_idx >= len(event1to2) - 1

    def get_time(self):
        return self.iter_idx

    def _start_wait(self):
        super(MockSequence, self)._start_wait()
        self.target_history.append(self.next_trial)

class TestSequence(unittest.TestCase):
    def setUp(self):
        self.targets = [1, 2, 3, 4, 5]
        self.exp = MockSequence(self.targets)

    def test_target_sequence(self):
        self.exp.run_sync()
        self.assertEqual(self.targets, self.exp.target_history)




###############################################################################
from riglib.hdfwriter import HDFWriter
import tables
import os
import numpy as np

test_output_fname = "test.hdf"

class TestHDFWriter(unittest.TestCase):
    def setUp(self):
        self.wr = wr = HDFWriter(test_output_fname)
        self.table1_dtype = np.dtype([("stuff", np.float64)])
        self.table2_dtype = np.dtype([("stuff2", np.float64), ("stuff3", np.uint8)])
        wr.register("table1", self.table1_dtype, include_msgs=True)
        wr.register("table2", self.table2_dtype, include_msgs=False)

        # send some data
        wr.send("table1", np.zeros(3, dtype=self.table1_dtype))
        wr.send("table1", np.ones(1, dtype=self.table1_dtype))
        wr.send("table2", np.ones(1, dtype=self.table2_dtype))
        wr.sendMsg("message!")
        wr.close()

    def test_h5_file_created(self):
        h5 = tables.open_file(test_output_fname)
        self.assertTrue(hasattr(h5, "root"))
        h5.close()

    def test_tables_exist(self):
        h5 = tables.open_file(test_output_fname)
        self.assertTrue(hasattr(h5.root, "table1"))
        self.assertTrue(hasattr(h5.root, "table1_msgs"))
        self.assertTrue(hasattr(h5.root, "table2"))
        self.assertFalse(hasattr(h5.root, "table2_msgs"))

        self.assertEqual(len(h5.root.table1), 4) # NOTE this only works after a bugfix in HDFWriter
        self.assertEqual(len(h5.root.table2), 1)
        self.assertEqual(len(h5.root.table1_msgs), 1)

        self.assertEqual(h5.root.table1_msgs[0]['msg'], "message!")
        self.assertEqual(h5.root.table1_msgs[0]['time'], 4)

        self.assertTrue(np.all(h5.root.table2[:]['stuff2'] == 1))
        h5.close()


    def tearDown(self):
        pass
        # if os.path.exists(test_output_fname):
        #     os.remove(test_output_fname)


###############################################################################
## Accumulators ###############################################################
from riglib.bmi import accumulator

class TestRectAccumulator(unittest.TestCase):
    def setUp(self):
        self.count_max = 10
        feature_shape = (2,)
        feature_dtype = np.float64
        self.accumulator = accumulator.RectWindowSpikeRateEstimator(self.count_max, feature_shape,
            feature_dtype)

    def test_acc(self):
        ones = np.ones(2,)
        for k in range(self.count_max + 1):
            feature_vec, decode_flag = self.accumulator(ones)
            if k < self.count_max - 1:
                self.assertTrue(np.all(feature_vec == k + 1))
                self.assertFalse(decode_flag)
            elif k == self.count_max - 1:
                self.assertTrue(np.all(feature_vec == k + 1))
                self.assertTrue(decode_flag)
            elif k > self.count_max - 1:
                self.assertTrue(np.all(feature_vec == k - (self.count_max - 1)))
                self.assertFalse(decode_flag)

class TestNullAccumulator(unittest.TestCase):
    def setUp(self):
        self.count_max = 10
        feature_shape = (2,)
        feature_dtype = np.float64
        self.accumulator = accumulator.NullAccumulator(self.count_max)

    def test_acc(self):
        ones = np.ones(2,)
        for k in range(self.count_max + 1):
            feature_vec, decode_flag = self.accumulator(ones)
            if k < self.count_max - 1:
                self.assertTrue(np.all(feature_vec == 1))
                self.assertFalse(decode_flag)
            elif k == self.count_max - 1:
                self.assertTrue(np.all(feature_vec == 1))
                self.assertTrue(decode_flag)
            elif k > self.count_max - 1:
                self.assertTrue(np.all(feature_vec == 1))
                self.assertFalse(decode_flag)

###############################################################################
## GaussianState ##############################################################
from riglib.bmi.bmi import GaussianState

class TestGaussianState(unittest.TestCase):
    def test_mul_by_scalara(self):
        x = GaussianState(np.zeros(2), np.diag(np.ones(2)))
        y = x * 2
        self.assertEqual(y.cov[0,0], 4)
        self.assertEqual(y.cov[1,1], 4)
        self.assertEqual(y.cov[0,1], 0)
        self.assertEqual(y.cov[1,0], 0)

    def test_mul_by_mat(self):
        x = GaussianState(np.zeros(2), np.diag(np.ones(2)))
        A = np.mat(np.diag([2, 3]))
        y = A * x
        self.assertEqual(y.cov[0,0], 4)
        self.assertEqual(y.cov[1,1], 9)
        self.assertEqual(y.cov[0,1], 0)
        self.assertEqual(y.cov[1,0], 0)

    def test_add(self):
        x = GaussianState(np.ones(2)*3, np.diag(np.ones(2)))
        y = GaussianState(np.ones(2)*4, np.diag(np.ones(2)))
        z = x + y

        self.assertEqual(z.mean[0,0], 7)
        self.assertEqual(z.mean[1,0], 7)

        self.assertEqual(z.cov[0,0], 2)
        self.assertEqual(z.cov[1,0], 0)

###############################################################################
## Kalman filter ##############################################################
from riglib.bmi.kfdecoder import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def test_kf_prediction(self):
        tol = 1e-10

        a = 0.9
        q = 2
        A = np.diag([a, a])
        W = np.diag([1.0, 1.0])
        C = np.array([[0.0, 1.0], [1.0, 0.0]])
        Q = np.diag([q, q])
        kf = KalmanFilter(A, W, C, Q)


        p = 1.0/a**2
        x_t = GaussianState(np.mat([0, 0]).reshape(-1,1), np.diag([p, p]))

        y = 0.1
        y_t = np.mat([y, -y]).reshape(-1,1)
        x_t_est = kf._forward_infer(x_t, y_t)

        K_expected = 0.5

        self.assertTrue(np.abs(x_t_est.mean[0,0] - y*K_expected*(-1)) < tol)
        self.assertTrue(np.abs(x_t_est.mean[1,0] - y*K_expected*(1)) < tol)

    def test_kf_train_obs_model_noiseless(self):
        tol = 1e-10
        X = np.array([1, 2, 3, 4, 5])
        slope = 0.5
        offset = 1.0
        Y = slope*X + offset

        C, Q = KalmanFilter.MLE_obs_model(X.reshape(1,-1), Y.reshape(1,-1))
        self.assertTrue(np.abs(C[0,0] - slope) < tol)
        self.assertTrue(np.abs(C[0,1] - offset) < tol)
        self.assertTrue(Q[0,0] < tol)

    def test_kf_train_state_space_model_noiseless(self):
        tol = 1e-10
        a = 0.9
        b = 0
        X = [1.0]
        for k in range(1, 5):
            X.append(X[k-1]*a + b)

        A, W = KalmanFilter.MLE_state_space_model(np.array(X).reshape(1,-1))

        self.assertTrue(np.abs(A[0, 0] - a) < tol)
        self.assertTrue(np.abs(A[1, 1] - 1) < tol) # offset "state"
        self.assertTrue(np.all(W < tol))


###############################################################################
## Kalman filter decoder ######################################################
from riglib.bmi.kfdecoder import KFDecoder
from riglib.bmi.state_space_models import State, StateSpace

class TestKFDecoder(unittest.TestCase):
    def test_kfdecoder_prediction(self):
        tol = 1e-10

        a = 0.9
        q = 2
        A = np.diag([a, a])
        W = np.diag([1.0, 1.0])
        C = np.array([[0.0, 1.0], [1.0, 0.0]]) # these are intentionally crossed
        Q = np.diag([q, q])
        kf = KalmanFilter(A, W, C, Q)

        units = [(1, 1), (2, 1)]
        ssm = StateSpace(State("state1", stochastic=True, drives_obs=True, order=0), State("state2", stochastic=True, drives_obs=True, order=0))

        decoder = KFDecoder(kf, units, ssm)
        p = 1.0/a**2

        y = 0.1
        y_t = np.array([y, -y])

        kf._init_state(np.array([0, 0]), np.diag([p, p]))
        x_t_est = decoder.predict(y_t)

        K_expected = 0.5

        self.assertTrue(np.abs(x_t_est[0] - y*K_expected*(-1)) < tol)
        self.assertTrue(np.abs(x_t_est[1] - y*K_expected*(1)) < tol)

###############################################################################
## Goal calculators ###########################################################
from riglib.bmi import goal_calculators, state_space_models
class TestZeroVelocityGoal(unittest.TestCase):
    def test_zero_velocity_goal(self):
        ssm = state_space_models.StateSpaceEndptVel2D()
        goal_calc = goal_calculators.ZeroVelocityGoal(ssm)

        target_pos = np.array([0, 0, 0], dtype=np.float64)
        (goal_state, error), _ = goal_calc(target_pos)

        self.assertTrue(np.array_equal(goal_state.ravel(), np.array([0, 0, 0, 0, 0, 0, 1])))




# /Users/sgowda/code/bmi3d/riglib/bmi/bmi.py                                578    469    19%   37, 43, 56-69, 80-81, 89-99, 123-124, 130, 150-157, 165-166, 195-212, 219, 226, 246-251, 271-276, 283-284, 287, 302-303, 306-312, 322-323, 329, 332-335, 366-392, 399-412, 430-454, 461, 474-483, 489-497, 514-528, 540-554, 560-578, 593-596, 603, 621-676, 691-696, 699-702, 709, 718, 733-734, 750-760, 773-777, 784, 812-820, 854-921, 938-962, 965, 974, 980-989, 996, 1003-1011, 1020, 1027-1036, 1043-1047, 1054-1056, 1063, 1081-1085, 1091-1092, 1110-1152, 1155, 1161, 1169, 1172-1176, 1179-1180, 1183-1184, 1190-1206, 1224-1271, 1274-1315

# /Users/sgowda/code/bmi3d/riglib/bmi/extractor.py                          648    648     0%   5-1367
# /Users/sgowda/code/bmi3d/riglib/bmi/feedback_controllers.py               115    115     0%   8-382
# /Users/sgowda/code/bmi3d/riglib/bmi/goal_calculators.py                   152    152     0%   2-356
# /Users/sgowda/code/bmi3d/riglib/bmi/kfdecoder.py                          428    428     0%   5-903
# /Users/sgowda/code/bmi3d/riglib/bmi/kfdecoder_fcns.py                     274    274     0%   2-412

# /Users/sgowda/code/bmi3d/riglib/bmi/robot_arms.py                         291    291     0%   8-721
# /Users/sgowda/code/bmi3d/riglib/bmi/sim_neurons.py                        415    415     0%   2-823
# /Users/sgowda/code/bmi3d/riglib/bmi/sskfdecoder.py                         59     59     0%   5-177
# /Users/sgowda/code/bmi3d/riglib/bmi/state_space_models.py                 205    205     0%   2-507
# /Users/sgowda/code/bmi3d/riglib/bmi/train.py                              705    705     0%   4-1412

# /Users/sgowda/code/bmi3d/riglib/bmi/ppfdecoder.py                         232    232     0%   5-510

# /Users/sgowda/code/bmi3d/riglib/bmi/assist.py                              47     47     0%   7-137
# /Users/sgowda/code/bmi3d/riglib/bmi/clda.py                               527    527     0%   7-1117        

if __name__ == '__main__':
    unittest.main()
