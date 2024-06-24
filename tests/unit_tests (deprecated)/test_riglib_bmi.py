import unittest
import numpy as np
from reqlib import swreq
from requirements import *

###############################################################################
## Kalman filter ##############################################################
from riglib.bmi.kfdecoder import KalmanFilter
from riglib.bmi.bmi import GaussianState

# 48, 59, 82-84, 101, 146, 149, 158-171, 206-242, 263-279, 298-307, 327-336, 347, 
# 353-357, 365-368, 397-417, 428-432, 440-454, 458-476, 486-515, 521-524, 530-532, 
# 537-601, 654-662, 668-672, 687-690, 706-707, 724-747, 760-785, 791-792, 808-810, 
# 817-907

class TestKalmanFilter(unittest.TestCase):
    @swreq(req_kf)
    def test_kf_prediction(self):
        """Single iteration of KF prediction shall match hand-verifable reference output."""
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

    @swreq(req_kf_mle)
    def test_kf_train_obs_model_noiseless(self):
        """MLE of KF observation model shall match known parameters used to generate sample ipnut data"""
        tol = 1e-10
        X = np.array([1, 2, 3, 4, 5])
        slope = 0.5
        offset = 1.0
        Y = slope*X + offset

        C, Q = KalmanFilter.MLE_obs_model(X.reshape(1,-1), Y.reshape(1,-1))
        self.assertTrue(np.abs(C[0,0] - slope) < tol)
        self.assertTrue(np.abs(C[0,1] - offset) < tol)
        self.assertTrue(Q[0,0] < tol)

    @swreq(req_kf_mle)
    def test_kf_train_state_space_model_noiseless(self):
        """MLE of KF state-space model shall match known parameters used to generate sample ipnut data"""
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
## Goal calculators ###########################################################
from riglib.bmi import goal_calculators, state_space_models
class TestZeroVelocityGoal(unittest.TestCase):
    def test_zero_velocity_goal(self):
        ssm = state_space_models.StateSpaceEndptVel2D()
        goal_calc = goal_calculators.ZeroVelocityGoal(ssm)

        target_pos = np.array([0, 0, 0], dtype=np.float64)
        (goal_state, error), _ = goal_calc(target_pos)

        self.assertTrue(np.array_equal(goal_state.ravel(), np.array([0, 0, 0, 0, 0, 0, 1])))

if __name__ == '__main__':
    unittest.main()