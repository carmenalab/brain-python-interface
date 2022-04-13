from .pyeCubeStream import eCubeStream
import numpy as np
import time

def gracefully_stop_session(ec, session, dch=25, timeout=10):
    '''
    Wait until a digital event occurs in the given channel, or until the timeout is reached, to stop recording.
    '''
    ec.add(('DigitalPanelAsChans',(dch,dch)))

    # Start streaming
    ec.start()
    t0 = time.perf_counter()

    # Wait for digital pulse
    received_stop_signal = False
    while True:
        data_block = ec.get() # in the form of (time_stamp, data_source, data_content)
        
        if time.perf_counter() - t0 > timeout:
            break

        values = data_block[2]
        if np.sum(values) > 0:
            received_stop_signal = True
            break

    # Stop streaming
    ec.stop()
    ec.remotestopsave(session)
    ec.remove(('DigitalPanelAsChans',))

    return received_stop_signal


def remove_headstage_sources(ec):
    '''
    Removes headstages sources so they can be added again to the same eCube instance later.
    '''
    sources = ec.listadded()
    if len(sources[0]) > 0:
        for hs in np.unique(sources[0][0]):
            ec.remove(('Headstages', int(hs)))