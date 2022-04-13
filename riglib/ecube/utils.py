from .pyeCubeStream import eCubeStream
import numpy as np
import time

def remove_headstage_sources(ec):
    '''
    Removes headstages sources so they can be added again to the same eCube instance later.
    '''
    sources = ec.listadded()
    if len(sources[0]) > 0:
        for hs in np.unique(sources[0][0]):
            ec.remove(('Headstages', int(hs)))