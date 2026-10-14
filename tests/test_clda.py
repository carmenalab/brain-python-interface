'''
Tests of the CLDA learners and updaters which do not need a rig or a task, only riglib.bmi
'''
from riglib.bmi import clda, wfdecoder, state_space_models, train, accumulator
from riglib.bmi.bmi import BMISystem
import numpy as np
import unittest


class TestPositionErrorLearner(unittest.TestCase):

    def setUp(self):
        self.ssm = state_space_models.StateSpaceEndptVel2D()
        self.learner = clda.PositionErrorLearner(10, gain=2., zero_tol=0.1)

    def test_intended_velocity(self):
        current = np.array([1., 0., 2., 5., 0., 5., 1.]).reshape(-1, 1)
        target = np.array([4., 0., -2., 0., 0., 0., 1.]).reshape(-1, 1)
        int_kin = self.learner.calc_int_kin(current, target, current, 'target', state_order=self.ssm.state_order)

        self.assertEqual(int_kin.shape, (7, 1))
        int_kin = np.asarray(int_kin).ravel()
        # velocity points from the cursor to the target, scaled by the gain
        np.testing.assert_allclose(int_kin[[3, 5]], [6., -8.])
        # the position, unused velocity and offset states are copied from the current state
        np.testing.assert_allclose(int_kin[[0, 1, 2, 4, 6]], [1., 0., 2., 0., 1.])

    def test_zero_tolerance(self):
        current = np.array([1., 0., 2., 5., 0., 5., 1.]).reshape(-1, 1)
        target = current.copy()
        target[0] += 0.05
        int_kin = self.learner.calc_int_kin(current, target, current, 'target', state_order=self.ssm.state_order)
        np.testing.assert_array_equal(np.asarray(int_kin).ravel()[[3, 5]], [0., 0.])

    def test_no_target(self):
        current = np.zeros((7, 1))
        target = np.ones((7, 1)) * np.nan
        self.assertIsNone(self.learner.calc_int_kin(current, target, current, 'no_target', state_order=self.ssm.state_order))

    def test_learn_states(self):
        learner = clda.PositionErrorLearner(10, gain=2., learn_states=['target'])
        current = np.zeros((7, 1))
        target = np.ones((7, 1))
        self.assertIsNotNone(learner.calc_int_kin(current, target, current, 'target', state_order=self.ssm.state_order))
        self.assertIsNone(learner.calc_int_kin(current, target, current, 'wait', state_order=self.ssm.state_order))

    def test_batch(self):
        current = np.zeros((7, 1))
        target = np.ones((7, 1))
        obs = np.ones((4, 1))
        for k in range(10):
            self.assertFalse(self.learner.is_ready())
            self.learner(obs*k, current, target, current, 'target', state_order=self.ssm.state_order)
        self.assertTrue(self.learner.is_ready())

        batch = self.learner.get_batch()
        self.assertEqual(batch['intended_kin'].shape, (7, 10))
        self.assertEqual(batch['spike_counts'].shape, (4, 10))
        np.testing.assert_array_equal(batch['spike_counts'][0, :], np.arange(10))
        self.assertFalse(self.learner.is_ready())


class TestWFSmoothbatch(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.ssm = state_space_models.StateSpaceEndptVel2D()
        self.n_units = 8
        self.n_taps = 3
        self.T = 300
        self.units = np.vstack([np.arange(1, self.n_units+1), np.zeros(self.n_units)]).T.astype(np.int32)

        # observations linearly related to the velocity, plus noise
        self.vel = np.random.randn(2, self.T)
        self.C = np.random.randn(self.n_units, 2)
        self.obs = self.C.dot(self.vel) + 3 + 0.1*np.random.randn(self.n_units, self.T)

        self.X = self.vel
        self.Y = wfdecoder.WienerFilter.form_obs_history(self.obs, n_taps=self.n_taps)

    def _make_decoder(self, seed_noise=0.):
        kin = np.zeros((7, self.T))
        kin[[3, 5], :] = self.vel
        decoder = train.train_WFDecoder_abstract(self.ssm, kin, self.obs, self.units, 0.1, n_taps=self.n_taps)
        decoder.filt.H = np.asmatrix(np.asarray(decoder.filt.H) + seed_noise*np.random.randn(*decoder.filt.H.shape))
        decoder.filt._init_state()
        return decoder

    def test_gradient(self):
        from scipy.optimize import check_grad
        mask = clda.WFSmoothbatch._penalty_mask(2, self.Y.shape[0])
        args = (self.X, self.Y, 0.3, 0.7, mask)
        H0 = np.random.randn(2 * self.Y.shape[0])
        err = check_grad(clda.WFSmoothbatch.cost_l2, clda.WFSmoothbatch.gradient_cost_l2, H0, *args)
        grad_norm = np.linalg.norm(clda.WFSmoothbatch.gradient_cost_l2(H0, *args))
        self.assertLess(err/grad_norm, 1e-5)

    def test_bfgs_matches_closed_form(self):
        H_bfgs, info = clda.WFSmoothbatch.estimate_filter(self.X, self.Y, lambda_E=0.1, lambda_D=0.1, solver='bfgs')
        H_exact, _ = clda.WFSmoothbatch.estimate_filter(self.X, self.Y, lambda_E=0.1, lambda_D=0.1, solver='exact')
        self.assertTrue(info['success'])
        np.testing.assert_allclose(H_bfgs, H_exact, atol=1e-5)

        # the closed form solution is a stationary point of the cost
        mask = clda.WFSmoothbatch._penalty_mask(2, self.Y.shape[0])
        grad = clda.WFSmoothbatch.gradient_cost_l2(H_exact, self.X, self.Y, 0.1, 0.1, mask)
        self.assertLess(np.abs(grad).max(), 1e-8)

    def test_no_regularization_is_least_squares(self):
        H_hat, _ = clda.WFSmoothbatch.estimate_filter(self.X, self.Y, lambda_E=1., lambda_D=0., solver='exact')
        H_lstsq = np.linalg.lstsq(self.Y.T, self.X.T, rcond=None)[0].T
        np.testing.assert_allclose(H_hat, H_lstsq, atol=1e-8)

        # ...and matches the training function of the decoder
        H_mle = wfdecoder.WienerFilter.MLE_filter(self.X, self.obs, n_taps=self.n_taps, regularizer=0.5)
        H_ridge, _ = clda.WFSmoothbatch.estimate_filter(self.X[:, self.n_taps-1:], self.Y[:, self.n_taps-1:], lambda_E=1., lambda_D=0.5, solver='exact')
        np.testing.assert_allclose(H_ridge, np.asarray(H_mle), atol=1e-8)

    def test_exact_rank_deficient(self):
        # a silent unit makes Y*Y^T singular without regularization
        Y = np.vstack([self.Y[:-1], np.zeros((1, self.T)), self.Y[-1:]])
        H_hat, info = clda.WFSmoothbatch.estimate_filter(self.X, Y, lambda_E=1., lambda_D=0., solver='exact')
        H_lstsq = np.linalg.lstsq(Y.T, self.X.T, rcond=None)[0].T
        np.testing.assert_allclose(H_hat, H_lstsq, atol=1e-8)

    def test_train_single_feature(self):
        kin = np.zeros((7, self.T))
        kin[[3, 5], :] = self.vel
        decoder = train.train_WFDecoder_abstract(self.ssm, kin, self.obs[:1], self.units[:1], 0.1, n_taps=self.n_taps)
        self.assertEqual(decoder.filt.n_features, 1)

    def test_mle_filter_no_offset(self):
        H = wfdecoder.WienerFilter.MLE_filter(self.X, self.obs, n_taps=self.n_taps, include_offset=False)
        self.assertEqual(H.shape, (2, self.n_units*self.n_taps + 1))
        np.testing.assert_array_equal(H[:, -1], 0)
        decoder = train.make_fixed_wf_decoder(self.units, self.ssm, H, n_taps=self.n_taps)
        decoder.filt._init_state()
        decoder.filt(self.obs[:, 0])

    def test_control_inputs(self):
        decoder = self._make_decoder()
        B = decoder.filt.B
        decoder.filt(self.obs[:, 0], u=np.zeros((B.shape[1], 1)))
        # without feedback gains x_target cannot be used
        x_target = np.zeros((self.ssm.n_states, 1))
        with self.assertRaises(ValueError):
            decoder.filt(self.obs[:, 0], x_target=x_target)
        decoder.filt(self.obs[:, 0], x_target=x_target, F=np.zeros((B.shape[1], self.ssm.n_states)))

        # a filter constructed without B rejects the control inputs explicitly
        filt = wfdecoder.WienerFilter(decoder.filt.A, decoder.filt.W, decoder.filt.H, n_taps=self.n_taps)
        filt._init_state()
        with self.assertRaises(ValueError):
            filt(self.obs[:, 0], u=np.zeros((B.shape[1], 1)))

    def test_regularization_shrinks_weights(self):
        H_ridge, _ = clda.WFSmoothbatch.estimate_filter(self.X, self.Y, lambda_E=1., lambda_D=100., solver='bfgs')
        H_lstsq, _ = clda.WFSmoothbatch.estimate_filter(self.X, self.Y, lambda_E=1., lambda_D=0., solver='exact')
        self.assertLess(np.linalg.norm(H_ridge[:, :-1]), np.linalg.norm(H_lstsq[:, :-1]))

    def test_smoothbatch_update(self):
        decoder = self._make_decoder(seed_noise=1.)
        H_old = np.asarray(decoder.filt.H).copy()

        # half life equal to the batch time gives a 50/50 blend of the old and new weights
        batch_time = self.T * decoder.binlen
        updater = clda.WFSmoothbatch(batch_time, batch_time, lambda_E=1., lambda_D=0.1, solver='exact')
        updater.init(decoder)
        self.assertAlmostEqual(updater.rho, 0.5)

        intended_kin = np.zeros((7, self.T))
        intended_kin[[3, 5], :] = self.vel
        new_params = updater.calc(intended_kin=intended_kin, spike_counts=self.obs, decoder=decoder)
        H_new = np.asarray(new_params['filt.H'])

        H_hat, _ = clda.WFSmoothbatch.estimate_filter(self.X[:, self.n_taps-1:], self.Y[:, self.n_taps-1:], lambda_E=1., lambda_D=0.1, solver='exact')
        np.testing.assert_allclose(H_new[[3, 5], :], 0.5*H_old[[3, 5], :] + 0.5*H_hat)

        # weights of the states not estimated from the observations are untouched
        np.testing.assert_array_equal(H_new[[0, 1, 2, 4, 6], :], H_old[[0, 1, 2, 4, 6], :])

        # a half-life passed at call time overrides the default step size
        new_params = updater.calc(intended_kin=intended_kin, spike_counts=self.obs, decoder=decoder, half_life=1e9)
        np.testing.assert_allclose(np.asarray(new_params['filt.H']), H_old, atol=1e-6)

    def test_zscore(self):
        kin = np.zeros((7, self.T))
        kin[[3, 5], :] = self.vel
        decoder = train.train_WFDecoder_abstract(self.ssm, kin, self.obs, self.units, 0.1, n_taps=self.n_taps, zscore=True)
        self.assertTrue(decoder.zscore)

        # with rho = 0, the update replaces the weights with the estimate from the (normalized) batch
        updater = clda.WFSmoothbatch(1., 1e-9, lambda_E=1., lambda_D=0., solver='exact')
        updater.init(decoder)
        new_params = updater.calc(intended_kin=kin, spike_counts=self.obs, decoder=decoder)
        np.testing.assert_allclose(np.asarray(new_params['filt.H']), np.asarray(decoder.filt.H), atol=1e-6)

    def test_short_batch(self):
        decoder = self._make_decoder()
        updater = clda.WFSmoothbatch(1., 1., solver='exact')
        updater.init(decoder)
        new_params = updater.calc(intended_kin=np.zeros((7, self.n_taps-1)), spike_counts=self.obs[:, :self.n_taps-1], decoder=decoder)
        np.testing.assert_array_equal(np.asarray(new_params['filt.H']), np.asarray(decoder.filt.H))

    def test_multiproc(self):
        import time
        decoder = self._make_decoder(seed_noise=1.)
        intended_kin = np.zeros((7, self.T))
        intended_kin[[3, 5], :] = self.vel

        updater_sync = clda.WFSmoothbatch(1., 1., lambda_E=0.1, lambda_D=0.1, solver='bfgs', multiproc=False)
        updater_sync.init(decoder)
        H_sync = updater_sync.calc(intended_kin=intended_kin, spike_counts=self.obs, decoder=decoder)['filt.H']

        # the same calculation in a separate process
        updater = clda.WFSmoothbatch(1., 1., lambda_E=0.1, lambda_D=0.1, solver='bfgs', multiproc=True)
        updater.init(decoder)
        updater(intended_kin=intended_kin, spike_counts=self.obs, decoder=decoder)
        self.assertIsNone(updater.get_result())
        result = None
        t0 = time.time()
        while result is None and time.time() - t0 < 20:
            time.sleep(0.1)
            result = updater.get_result()
        self.assertIsNotNone(result, "no result from the updater process")
        np.testing.assert_allclose(np.asarray(result['filt.H']), np.asarray(H_sync), atol=1e-6)

        # the result arrives on a later cycle of the BMI system, which must still log the batch
        bmi_system = BMISystem(decoder, clda.PositionErrorLearner(5, gain=2.), updater, accumulator.NullAccumulator(1))
        target_state = np.zeros((7, 1))
        target_state[0, 0] = 5
        target_state[6, 0] = 1
        obs = np.ones((self.n_units, 1))
        t0 = time.time()
        update_flag = False
        n_calls = 0
        while not update_flag and time.time() - t0 < 20:
            _, update_flag = bmi_system(obs, target_state, 'target', learn_flag=True)
            n_calls += 1
            time.sleep(0.01)
        self.assertTrue(update_flag)
        self.assertGreater(n_calls, 5)
        self.assertEqual(len(bmi_system.param_hist), 1)
        self.assertEqual(bmi_system.param_hist[0]['intended_kin'].shape, (7, 5))
        self.assertTrue(bmi_system.learner.enabled)

        updater.calculator.stop()
        updater.calculator.join(timeout=5)
        self.assertFalse(updater.calculator.is_alive())

    def test_wrong_decoder(self):
        from riglib.bmi import kfdecoder
        C = np.zeros([self.n_units, 7])
        decoder = train.make_fixed_kf_decoder(self.units, self.ssm, C, dt=0.1)
        updater = clda.WFSmoothbatch(1., 1.)
        with self.assertRaises(TypeError):
            updater.init(decoder)

    def test_closed_loop_adaptation(self):
        '''
        Simulate a subject whose neural features encode the intended velocity toward the target, and
        check that CLDA improves a poorly seeded decoder in closed loop
        '''
        np.random.seed(1)
        decoder = self._make_decoder(seed_noise=1.)
        decoder.set_call_rate(10.)

        gain = 2.
        batch_time = 5.
        learner = clda.PositionErrorLearner(int(batch_time/decoder.binlen), gain=gain)
        updater = clda.WFSmoothbatch(batch_time, batch_time, lambda_E=0.1, lambda_D=0.1, solver='bfgs')
        bmi_system = BMISystem(decoder, learner, updater, accumulator.NullAccumulator(1))

        targets = 6 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1]]) / 1.

        def run_block(n_trials, learn_flag):
            dists = []
            n_updates = 0
            for trial in range(n_trials):
                target_state = np.zeros((7, 1))
                target_state[[0, 2], 0] = targets[trial % len(targets)]
                target_state[6, 0] = 1
                decoder.filt._init_state()
                for t in range(50):
                    state = np.asarray(decoder.get_state()).ravel()
                    v_int = gain*(target_state[[0, 2], 0] - state[[0, 2]])
                    obs = self.C.dot(v_int) + 3 + 0.1*np.random.randn(self.n_units)
                    _, update_flag = bmi_system(obs.reshape(-1, 1), target_state, 'target', learn_flag=learn_flag)
                    n_updates += update_flag
                state = np.asarray(decoder.get_state()).ravel()
                dists.append(np.linalg.norm(state[[0, 2]] - target_state[[0, 2], 0]))
            return np.mean(dists), n_updates

        dist_before, _ = run_block(6, False)
        _, n_updates = run_block(24, True)
        dist_after, _ = run_block(6, False)

        self.assertGreater(n_updates, 10)
        self.assertLess(dist_after, 0.5*dist_before)
        self.assertLess(dist_after, 1.)
        self.assertEqual(len(bmi_system.param_hist), n_updates)


if __name__ == '__main__':
    unittest.main()
