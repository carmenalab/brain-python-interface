from db import dbfunctions
from db import json_param
from db.tracker import models

from riglib import experiment
from features.generator_features import Autostart
from features.hdf_features import SaveHDF
from features.phasespace_features import MotionData

from features.plexon_features import PlexonBMI
from riglib.experiment import generate

from analysis import performance
from tasks.manualcontrolmultitasks import ManualControlMulti
from tasks.bmimultitasks import SimBMIControlMulti
import riglib.bmi.onedim_lfp_decoder as old
from riglib.bmi import clda
from riglib.bmi import train
from analysis import performance



import os
os.environ['DISPLAY'] = ':0'

save = True

#task = models.Task.objects.get(name='lfp_mod')
task = models.Task.objects.get(name='lfp_mod_mc_reach_out')
#task = models.Task.objects.get(name='manual_control_multi')

base_class = task.get()

feats = [SaveHDF, Autostart, PlexonBMI, MotionData]
Exp = experiment.make(base_class, feats=feats)

#params.trait_norm(Exp.class_traits())
params = dict(session_length=10, plant_visible=True, lfp_plant_type='cursor_onedimLFP', mc_plant_type='cursor_14x14',
        rand_start=(0.,0.), max_tries=1)

gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)
exp = Exp(gen, **params)

import pickle
#decoder = pickle.load(open('/storage/decoders/cart20141206_06_test_lfp1d2.pkl'))
d#ecoder = pickle.load(open('/storage/decoders/cart20141208_12_test_PK.pkl'))
decoder = pickle.load(open('/storage/decoders/cart20141209_08_cart_2015_pilot_2.pkl'))
exp.decoder = decoder

exp.init()

exp.run()



def test_decoder():
    def _get_band_ind(freq_pts, band, band_set):
        band_ind = -1
        for b, bd in enumerate(band_set):
            if (bd[0]==band[0]) and (bd[1]==band[1]):
                band_ind = b
        if band_ind == -1:
            band_ind = b+1
            band_set.extend([band])

        fft_ind = dict()
        for band_idx, band in enumerate(band_set):
            fft_ind[band_idx] = [freq_idx for freq_idx, freq in enumerate(freq_pts) if band[0] <= freq < band[1]]

        return band_ind, band_set, fft_ind

    ## Test decoder: 
    lfp_control_band = [25, 40]
    lfp_totalpw_band = [1, 100]

    decoder = pickle.load(open('/storage/decoders/cart20141209_07_pk3.pkl'))

    decoder.filt.control_band_ind, decoder.extractor_kwargs['bands'], decoder.extractor_kwargs['fft_inds'] = \
        _get_band_ind(decoder.extractor_kwargs['fft_freqs'], lfp_control_band, decoder.extractor_kwargs['bands'])

    decoder.filt.totalpw_band_ind, decoder.extractor_kwargs['bands'], decoder.extractor_kwargs['fft_inds'] = \
        _get_band_ind(decoder.extractor_kwargs['fft_freqs'], lfp_totalpw_band, decoder.extractor_kwargs['bands'])

    decoder.filt.frac_lims = [0, .35]
    decoder.filt.powercap = 50
    decoder.filt.zboundaries = (-12, 12)
    decoder.extractor_kwargs['no_log'] = True
    decoder.extractor_kwargs['no_mean'] = True


    lfp = decoder.extractor_cls(None,**decoder.extractor_kwargs)
    for i in range(100):
        #lfp = np.random.random((45,1))
        dat = lfp(i)
        decoder.predict(dat['lfp_power'])

