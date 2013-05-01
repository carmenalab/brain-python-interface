# coding: utf-8
from scipy import signal
filtCutoff = 2.0
norm_pass = filtCutoff/ ((1000/bin)/2)
signal.butter(2, norm_pass, btype='low', analog=0, output='ba')
b, a = signal.butter(2, norm_pass, btype='low', analog=0, output='ba')
