from cbpyFiles import *
import matplotlib.pyplot as plt

fileName = '/Users/Blackrock/Desktop/Commonly Used Files and Folders/Sample Data/Sample Data/Great Sample Data By Nick/Ch1_4SpikeCh5-8SpikeContCh9-12ContAn15SpikeContAn16Cont001.ns5'

ns = nsxReader(fileName)

ns.readHeaders()
ns.readDataHeader()
ns.readData()

