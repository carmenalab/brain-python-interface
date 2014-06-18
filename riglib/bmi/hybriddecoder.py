'''
Classes for a "hybrid" decoder which is a combination of discrete and continuous state elements
'''

import numpy as np
import kfdecoder

class KalmanFilterWithReset(kfdecoder.KalmanFilter):
    '''    Docstring    '''
    def __init__(self, A, W, C, Q, clf, threshold, reset_state, is_stochastic=None):    
        '''    Docstring    '''
        super(KalmanFilterWithReset, self).__init__(A, W, C, Q, is_stochastic=None)
        self.clf = clf
        self.threshold = threshold
        self.reset_state = np.mat(reset_state).reshape(-1,1)

    @classmethod
    def create_from_kf(cls, kf, clf, threshold, reset_state):
        '''    Docstring    '''
        obj = KalmanFilterWithReset(kf.A, kf.W, kf.C, kf.Q, clf, threshold, reset_state, is_stochastic=kf.is_stochastic)
        obj.clf = clf
        obj.threshold = threshold   
        return obj

    def __getstate__(self):
        '''    Docstring    '''
        data = super(KalmanFilterWithReset, self).__getstate__()
        data['clf'] = self.clf
        data['threshold'] = self.threshold
        data['reset_state'] = self.reset_state
        return data

    def __setstate__(self, state):
        '''    Docstring    '''
        self.clf = state['clf']
        self.threshold = state['threshold']
        self.reset_state = state['reset_state']
        super(KalmanFilterWithReset, self).__setstate__(state)

    def _pickle_init(self):
        '''    Docstring    '''
        super(KalmanFilterWithReset, self)._pickle_init()
        self.trying_prob = np.nan

        import assist, robot_arms, train
        pi = np.pi
        ## TODO REMOVE HARDOCDING below
        ssm = train.tentacle_2D_state_space
        shoulder_anchor = np.array([2., 0., -15])        
        link_lengths = [15., 15., 5., 5.]
        kin_chain = robot_arms.PlanarXZKinematicChain(link_lengths, base_loc=shoulder_anchor)
        kin_chain.joint_limits = [(-pi,pi), (-pi,0), (-pi/2,pi/2), (-pi/2, 10*pi/180)]
        ### ENDHARDOCDING

        self.resetter = assist.TentacleAssist(ssm=ssm, kin_chain=kin_chain)
        
    def __call__(self, obs, testing=False, **kwargs):
        """
        Docstring 
        
        When the object is called directly, it's a wrapper for the 
        1-step forward inference function.
        """

        ## Run the classifier
        class_prob_t = self.clf.predict_proba(np.array(obs).ravel())
        alpha = 0.975
        if np.isnan(self.trying_prob): #len(lpf_class_prob) == 0:
            self.trying_prob = class_prob_t[0,1]
        else:
            self.trying_prob = alpha*self.trying_prob + (1-alpha)*class_prob_t[0,1]
            
        if self.trying_prob > self.threshold:
            ## Operate KF as normal
            self.state = self._forward_infer(self.state, obs, **kwargs)
        else:
            ## Drive state toward the specified reset state by hijacking the "assist" forcing term input
            print 'driving toward init state!'
            if 'Bu' in kwargs: kwargs.pop('Bu')
            if 'assist_level' in kwargs: kwargs.pop('assist_level')
            Bu, assist_weight = self.resetter(self.state.mean, self.reset_state, 0.3, mode='target')            
            self.state = self._forward_infer(self.state, obs, Bu=Bu, **kwargs)

        if testing:
            return self.state.mean, self.trying_prob
        else:
            return self.state.mean    
