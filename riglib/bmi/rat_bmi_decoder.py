
from ..bmi import extractor
import numpy as np

from ..bmi import state_space_models
import datetime
import matplotlib.pyplot as plt

class State(object):
    '''For compatibility with other BMI decoding implementations, literally just holds the state'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class RatFilter(object):
    '''Moving Avergae Filter used in 1D or 2D LFP control:
    x_{t} = a0*x_{t} + a1*x_{t-1} + a2*x_{t-2} + ...
    x_{t} = b_1:t*x_{}


    Parameters

    ----------
    A: np.array of shape (N, )
        Weights for previous states
    X: np. array of previous states (N, )
    '''

    def __init__(self, task_params):
        self.e1_inds = task_params['e1_inds']
        self.e2_inds = task_params['e2_inds']
        self.FR_to_freq_fn = task_params['FR_to_freq_fn']
        self.t1 = task_params['t1']
        self.t2 = task_params['t2']
        self.mid = task_params['mid']
        self.dec_params = task_params

        #Cursor data (X)
        self.X = 0.

        #Freq data(F) 
        self.F = 0.
        self.baseline = False

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def init_from_task(self, n_units, **kwargs):
        #Define n_steps
        if 'nsteps' in kwargs:
            self.n_steps = kwargs['nsteps']
            self.A = np.ones(( self.n_steps, ))/float(self.n_steps)

            #Neural data (Y)
            self.Y = np.zeros(( self.n_steps, n_units))
            self.n_units = n_units

        else:
            raise Exception


    def _init_state(self, init_state=None,**kwargs):
        if init_state is None:
            init_state = 0

        self.state = State(init_state)

    def __call__(self, obs, **kwargs):
        self.state = self._mov_avg(obs, **kwargs)

    def _mov_avg(self, obs,**kwargs):
        ''' Function to compute moving average with old mean and new observation'''

        self.Y[:-1, :] = self.Y[1:, :]
        self.Y[-1, :] = np.squeeze(obs)

        d_fr = np.sum(self.Y[:, self.e1_inds], axis=1) - np.sum(self.Y[:, self.e2_inds], axis=1)
        mean_FR = np.dot(d_fr, self.A)
        self.X = mean_FR
        self.F = self.FR_to_freq_fn(self.X)
        return State(self.X)

    def FR_to_freq(self, mean_FR):
        return self.FR_to_freq_fn(mean_FR)

    def _pickle_init(self):
        pass

class IsmoreSleepFilter(RatFilter):
    def __init__(self, task_params):
        self.e1_inds = task_params['e1_inds']
        self.e2_inds = task_params['e2_inds']
        self.FR_to_alpha_fn = task_params['FR_to_alpha_fn']
        self.dec_params = task_params
        self.mid = task_params['mid']
        self.e1_max = task_params['e1_perc']
        self.e2_max = task_params['e2_perc']
        self.freq_lim = task_params['freq_lim']

        #Cursor data (X)
        self.FR = 0.

        #Freq data(F) 
        self.alpha = 0.
        self.model_attrs = []
        self.baseline = False

    def init_from_task(self, **kwargs):
        #Define n_steps
        self.n_steps = kwargs.pop('nsteps', 1)
        self.A = np.ones(( self.n_steps, ))/float(self.n_steps)

        #Neural data (Y)
        self.Y = np.zeros(( self.n_steps, len(self.e1_inds)+len(self.e2_inds)))
        self.n_units = len(self.e1_inds)+len(self.e2_inds)

    def _mov_avg(self, obs,**kwargs):
        ''' Function to compute moving average with old mean and new observation'''

        self.Y[:-1, :] = self.Y[1:, :]
        self.Y[-1, :] = np.squeeze(obs)

        if self.e1_max is not None:
            e1_tmp = np.min([self.e1_max, np.sum(self.Y[:, self.e1_inds], axis=1)])
        else:
            e1_tmp = np.sum(self.Y[:, self.e1_inds], axis=1)

        if self.e2_max is not None:
            e2_tmp = np.min([self.e2_max, np.sum(self.Y[:, self.e2_inds], axis=1)])
        else:
            e2_tmp = np.sum(self.Y[:, self.e2_inds], axis=1)

        d_fr = e1_tmp - e2_tmp
        mean_FR = np.dot(d_fr, self.A)
        self.FR = mean_FR
        self.alpha = self.FR_to_alpha_fn(self.FR)

        # Max alpha is -1 or 1 : 
        if self.alpha > self.freq_lim[1]:
            self.alpha = self.freq_lim[1]
        elif self.alpha < self.freq_lim[0]:
            self.alpha = self.freq_lim[0]

        self.baseline = self.FR < self.mid
        return State(self.alpha)

from riglib.bmi.bmi import Decoder
class RatDecoder(Decoder):

    def __init__(self, *args, **kwargs):
        
        #Args: filter, units, ssm, extractor_cls, extractor_kwargs
        super(RatDecoder, self).__init__(args[0], args[1], args[2])
        self.extractor_cls = args[3]
        self.extractor_kwargs = args[4]
        self.n_features = len(self.filt.e1_inds) + len(self.filt.e2_inds)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self,key,value)

    def predict(self, neural_obs, **kwargs):
        self.filt(neural_obs, **kwargs)


    def init_from_task(self,**kwargs):
        pass

class IsmoreSleepDecoder(RatDecoder):
    def __init__(self, *args, **kwargs):
        self.binlen = 0.1
        super(IsmoreSleepDecoder, self).__init__(*args, **kwargs)

########## Functions to make decoder ###########

def create_decoder(ssm, task_params):
    filter_ = RatFilter(task_params)
    decoder = RatDecoder(filter_, task_params['units'], ssm, task_params['extractor_cls'], dict())
    return decoder

########### Called from trainbmi.py to make decoder from Baseline #####
import re
cellname = re.compile(r'(\d{1,3})\s*(\w{1})')

def calc_decoder_from_baseline_file(neural_features, neural_features_unbinned, units, nsteps, prob_t1,
    prob_t2, timeout, timeout_pause, freq_lim, e1_inds, e2_inds, sim_fcn='rat', **kwargs):

    #Enter e1, e2 as string: 
    if np.logical_or(e1_inds is None, e2_inds is None):
        e1_string = input('Enter e1 cells: ')
        e2_string = input('Enter e2 cells: ')
        
        e1 = np.array([ (int(c), ord(u) - 96) for c, u in cellname.findall(e1_string)])
        e2 = np.array([ (int(c), ord(u) - 96) for c, u in cellname.findall(e2_string)])

        e1_inds = np.array([i for i, u in enumerate(units) if np.logical_and(u[0] in e1[:,0], u[1] in e1[:,1])])
        e2_inds = np.array([i for i, u in enumerate(units) if np.logical_and(u[0] in e2[:,0], u[1] in e2[:,1])])

    T = neural_features.shape[0]
    if 'saturate_perc' in kwargs:
        baseline_data = np.zeros((T - nsteps, 2))
    else:
        baseline_data = np.zeros((T - nsteps))
    for ib in range(T-nsteps):
        if 'saturate_perc' in kwargs:
            baseline_data[ib, 0] = np.mean(np.sum(neural_features[ib:ib+nsteps, 
                e1_inds], axis=1))
            baseline_data[ib, 1] = np.mean(np.sum(neural_features[ib:ib+nsteps, 
                e2_inds], axis=1)) 
        else:
            baseline_data[ib] = np.mean(np.sum(neural_features[ib:ib+nsteps, 
                e1_inds], axis=1))-np.mean(np.sum(neural_features[ib:ib+nsteps, 
                e2_inds], axis=1))

    if 'saturate_perc' in kwargs:

        sat_perc = kwargs.pop('saturate_perc')
        # ignore the first second of data
        e1_perc = np.percentile(baseline_data[20:, 0], sat_perc)
        e2_perc = np.percentile(baseline_data[20:, 1], sat_perc)

        baseline_data[:, 0][baseline_data[:, 0] > e1_perc] = e1_perc
        baseline_data[:, 1][baseline_data[:, 1] > e2_perc] = e2_perc
        baseline_data = baseline_data[:, 0] - baseline_data[:, 1]
    else:
        e1_perc = None
        e2_perc = None

    x, pdf, pdf_individual = generate_gmm(baseline_data)

    if sim_fcn == 'rat':
        t2, mid, t1, num_t1, num_t2, num_miss, FR_to_freq_fn = sim_data(x, pdf, pdf_individual, prob_t1, prob_t2,
        baseline_data, timeout, timeout_pause, freq_lim, sim_bmi_fcn='rat')

        return e1_inds, e2_inds, FR_to_freq_fn, units, t1, t2, mid
    
    elif sim_fcn == 'ismore':
        # Get fcn: 
        t1 = prob_under_pdf(x, pdf, prob_t1)
        t2 = prob_under_pdf(x, pdf, prob_t2)
        idx_mid = np.argmax(pdf)
        mid = x[idx_mid]
        FR_to_alpha_fn = map_to_freq(t2, mid, t1, freq_lim[0], freq_lim[1])
        
        # Plot FR to alpha fcn
        x_axis = np.linspace(t2, t1, 100)
        y_axis = []
        for xi in x_axis:
            yi = FR_to_alpha_fn(xi)
            yi = np.max([freq_lim[0], yi])
            yi = np.min([freq_lim[1], yi])
            y_axis.append(yi)

        x_axis2 = np.arange(-10, 10)
        y_axis2 = []
        for xi in x_axis2:
            y_axis2.append(FR_to_alpha_fn(xi))

        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.plot(x_axis, y_axis)
        ax.plot(x_axis2, y_axis2, 'r.')
        ax.plot([-5, 5], [0, 0], 'r--')

        kwargs2 = dict(replay_neural_features = neural_features, e1_inds=e1_inds, 
            e2_inds = e2_inds, FR_to_alpha_fn=FR_to_alpha_fn, mid = mid, e1_perc=e1_perc,
            e2_perc=e2_perc, freq_lim=freq_lim)

        filt = IsmoreSleepFilter(kwargs2)
        decoder = IsmoreSleepDecoder(filt, units, state_space_models.StateSpaceEndptPos1D(),
            extractor.BinnedSpikeCountsExtractor, {})
        if kwargs['skip_sim']:
            nrewards = []
            import pdb
            pdb.set_trace()

        else:
            if type(kwargs['targets_matrix']) is str:
                import pickle
                kwargs['targets_matrix'] = pickle.load(open(kwargs['targets_matrix']))
            pname = ismore_sim_bmi(neural_features_unbinned, decoder, targets_matrix=kwargs['targets_matrix'],
                session_length=kwargs['session_length'])

            # Analyze data: 
            import tables
            import matplotlib.pyplot as plt
            hdf = tables.openFile(pname[:-4]+'.hdf')

            # Plot x/y trajectory: 
            f, ax = plt.subplots()
            ax.plot(hdf.root.task[2:]['plant_pos'][:, 0], hdf.root.task[2:]['plant_pos'][:, 1])

            ix = np.nonzero(hdf.root.task[:]['target_index'] == 1)[0]
            ax.plot(hdf.root.task[ix[0]]['target_pos'][0], hdf.root.task[ix[0]]['target_pos'][1], 'r.',
                markersize=20)
        
            ix = np.nonzero(hdf.root.task[:]['target_index'] == 0)[0] + 5
            ax.plot(hdf.root.task[ix[0]]['target_pos'][0], hdf.root.task[ix[0]]['target_pos'][1], 'g.',
                markersize=20)

            nrewards = np.nonzero(hdf.root.task_msgs[:]['msg']=='reward')[0]

        return decoder, len(nrewards)


###### From Rat BMI #######
###### From Rat BMI #######
###### From Rat BMI #######
###### From Rat BMI #######

from sklearn import metrics
from sklearn.mixture import GMM
import numpy as np
import matplotlib.pyplot as plt

def generate_gmm(data, ax=None):
    ##reshape the data
    X = data.reshape(data.shape[0], 1)
    ##fit models with 1-10 components
    N = np.arange(1,11)
    models = [None for i in range(len(N))]
    for i in range(len(N)):
        models[i] = GMM(N[i]).fit(X)
    ##compute AIC
    AIC = [m.aic(X) for m in models]
    ##figure out the best-fit mixture
    M_best = models[np.argmin(AIC)]
    x = np.linspace(data.min()-1, data.max()+1, data.size)
    ##compute the pdf
    logprob, responsibilities = M_best.score_samples(x.reshape(x.size, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    #plot the stuff
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(X, 50, normed = True, histtype = 'stepfilled', alpha = 0.4)
    ax.plot(x, pdf, '-k')
    ax.plot(x, pdf_individual, '--k')
    ax.text(0.04, 0.96, "Best-fit Mixture",
        ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p(x)$')
    return x, pdf, pdf_individual

##this function takes in an array of x-values and an array
##of y-values that correspond to a probability density function
##and determines the x-value at which the area under the PDF is approximately
##equal to some value passed in the arguments.
def prob_under_pdf(x_pdf, y_pdf, prob):
    auc = 0
    i = 2
    while auc < prob:
        x_range = x_pdf[0:i]
        y_range = y_pdf[0:i]
        auc = metrics.auc(x_range, y_range)
        i+=1
    return x_pdf[i]

##function to map ensemble values to frequency values
def map_to_freq(t2, mid, t1, min_freq, max_freq):
    fr_pts = np.array([t2, mid, t1])
    freq_pts = np.array([min_freq, (float(max_freq) + float(min_freq))/2., max_freq])
    z = np.polyfit(fr_pts, freq_pts, 2)
    p = np.poly1d(z)
    return p

def sim_data(x, pdf, pdf_individual, prob_t1, prob_t2, data, 
    timeout, timeout_pause, freq_lim, sim_bmi_fcn='rat'):
    t1 = prob_under_pdf(x, pdf, prob_t1)
    t2 = prob_under_pdf(x, pdf, prob_t2)
    idx_mid = np.argmax(pdf)
    mid = x[idx_mid]
    fig, ax1 = plt.subplots()
    ax1.hist(data+np.random.normal(0, 0.1*data.std(), data.size), 50, 
        normed = True, histtype = 'stepfilled', alpha = 0.4)
    ax1.plot(x, pdf, '-k')
    ax1.plot(x, pdf_individual, '--k')
    ax1.text(0.04, 0.96, "Best-fit Mixture",
        ha='left', va='top', transform=ax1.transAxes)
    ax1.set_xlabel('Cursor Value (E1-E2)')
    ax1.set_ylabel('$p(x)$')
    ##find the points where t1 and t2 lie on the gaussian
    idx_t2 = np.where(x>t2)[0][0]
    x_t2 = t2
    y_t2 = pdf[idx_t2]
    idx_t1 = np.where(x>t1)[0][0]
    x_t1 = t1
    y_t1 = pdf[idx_t1]
    y_mid = pdf[idx_mid]
    ax1.plot(x_t1, y_t1, 'o', color = 'g')
    ax1.plot(x_t2, y_t2, 'o', color = 'g')
    ax1.plot(mid, y_mid, 'o', color = 'g')
    ax1.set_title("Firing rate histogram and gaussian fit")
    ax1.annotate('T1: ('+str(round(x_t1, 3))+')', xy=(x_t1, y_t1), xytext=(40,20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ax1.annotate('T2: ('+str(round(x_t2, 3))+')', xy=(x_t2, y_t2), xytext=(-40,20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ax1.annotate('Base: ('+str(round(mid, 3))+')', xy=(mid, y_mid), xytext=(-100,-20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ##get the control function
    p = map_to_freq(t2, mid, t1, freq_lim[0], freq_lim[1])
    ##run a simulation
    if sim_bmi_fcn=='rat':
        num_t1, num_t2, num_miss = sim_bmi(data, t1, t2, mid, timeout, timeout_pause, p)
    elif sim_bmi_fcn == 'ismore':
        num_t1, num_t2, num_miss = ismore_sim_bmi(data, t1, t2, mid, timeout, timeout_pause, p)
    
    print("Simulation results:\nNumber of T1: " + str(num_t1) + "\nNumber of T2: " + str(num_t2) + "\nNumber of Misses: " + str(num_miss))
    print("Calculated T2 value is " + str(round(t2, 5)))
    print("Calculated mid value is " + str(round(mid, 5)))
    print("Calculated T1 value is " + str(round(t1, 5)))
    ##plot the control functio
    plot_cursor_func(t2, mid, t1, freq_lim[0], freq_lim[1])
    #plt.show()
    return t2, mid, t1, num_t1, num_t2, num_miss, p

def sim_bmi(baseline_data, t1, t2, midpoint, timeout, timeout_pause, p):
    data = baseline_data 
    samp_int = 100. #ms

    ##get the timeout duration in samples
    timeout_samps = int((timeout*1000.0)/samp_int)
    timeout_pause_samps = int((timeout_pause*1000.0)/samp_int)
    ##"global" variables
    num_t1 = 0
    num_t2 = 0
    num_miss = 0
    back_to_baseline = 1
    ##run through the data and simulate BMI
    i = 0
    clock = 0
    while i < (data.shape[0]-1):
        cursor = data[i]
        ##check for a target hit
        if cursor >= t1:
            num_t1+=1
            i += int(4000/samp_int)
            back_to_baseline = 0
            ##wait for a return to baseline
            while cursor >= midpoint and i < (data.shape[0]-1):
                #advance the sample
                i+=1
                ##get cursor value
                cursor = data[i]
            ##reset the clock
            clock = 0
        elif cursor <= t2:
            num_t2+=1
            i += int(4000/samp_int)
            back_to_baseline = 0
            ##wait for a return to baseline
            while cursor >= midpoint and i < (data.shape[0]-1):
                #advance the sample
                i+=1
                ##get cursor value
                cursor = data[i]
            ##reset the clock
            clock = 0
        elif clock >= timeout_samps:
            ##advance the samples for the timeout duration
            i+= timeout_pause_samps
            num_miss += 1
            ##reset the clock
            clock = 0
        else:
            ##if nothing else, advance the clock and the sample
            i+= 1
            clock+=1
    return num_t1, num_t2, num_miss

def ismore_sim_bmi(baseline_data, decoder, targets_matrix=None, session_length=0.):
    import ismore.invasive.bmi_ismoretasks as bmi_ismoretasks
    from riglib import experiment
    from features.hdf_features import SaveHDF
    from ismore.brainamp_features import SimBrainAmpData
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    from features.blackrock_features import BlackrockBMI
    from ismore.exo_3D_visualization import Exo3DVisualizationInvasive

    targets = bmi_ismoretasks.SimBMIControlReplayFile.sleep_gen(length=100)
    plant_type = 'IsMore'
    kwargs=dict(session_length=session_length, replay_neural_features=baseline_data, decoder=decoder)
    
    if targets_matrix is not None:
        kwargs['targets_matrix']=targets_matrix
    
    Task = experiment.make(bmi_ismoretasks.SimBMIControlReplayFile, [SaveHDF])#, Exo3DVisualizationInvasive])
    task = Task(targets, plant_type=plant_type, **kwargs)
    task.run_sync()
    pnm = save_dec_enc(task)
    return pnm

def plot_cursor_func(t2, mid, t1, min_freq, max_freq):
    f, ax2 = plt.subplots()
    x = np.linspace(t2-1, t1+1, 1000)
    func = map_to_freq(t2, mid, t1, min_freq, max_freq)
    #fig, ax = plt.subplots()
    ax2.plot(t2, min_freq, 'o', color = 'r')
    ax2.plot(mid, np.floor((max_freq-min_freq)/2), 'o', color = 'r')
    ax2.plot(t1, max_freq, 'o', color = 'r')
    ax2.plot(x, func(x), '-', color = 'g')
    ax2.annotate('T1: ('+str(round(t1, 3))+')', xy=(t1, max_freq), xytext=(-20, 20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ax2.annotate('T2: ('+str(round(t2, 3))+')', xy=(t2, min_freq), xytext=(-20,20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ax2.annotate('Base: ('+str(round(mid, 3))+')', xy=(mid, np.floor((max_freq-min_freq)/2)), xytext=(-20,20), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))
    ax2.set_ylabel("Feedback frequency")
    ax2.set_xlabel("Cursor value (E1-E2)")
    ax2.set_title("Cursor-frequency map", fontsize = 18)

def save_dec_enc(task, pref='sleep_sim_'):
    '''
    Summary: method to save encoder / decoder and hdf file information from task in sim_data folder
    Input param: task: task, output from arm_assist_main, or generally task object
    Input param: pref: prefix to saved file names (defaults to 'enc' for encoder)
    Output param: pkl file name used to save encoder/decoder
    '''
    #enc = task.encoder
    task.decoder.save()
    #enc.corresp_dec = task.decoder

    #Save task info
    import pickle
    ct = datetime.datetime.now()
    #pnm = '/Users/preeyakhanna/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%Y%m%d_%H_%M_%S") + '.pkl'
    pnm = '/home/tecnalia/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    pnm2 = '/Users/preeyakhanna/code/ismore/ismore_tests/sim_data/'+pref + ct.strftime("%m%d%y_%H%M") + '.pkl'
    
    try:
        pickle.dump(dict(), open(pnm,'wb'))
    except:
        pickle.dump(dict(), open(pnm2, 'wb'))
        pnm = pnm2

    #Save HDF file
    new_hdf = pnm[:-4]+'.hdf'
    import shutil
    f = open(task.h5file.name)
    f.close()

    #Wait 
    import time
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    import time
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

    #Return filename
    return pnm