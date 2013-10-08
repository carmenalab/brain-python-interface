from riglib.bmi import train

test_mc = True
test_bc = False
if test_mc:
    block = 'cart20130815_01'
    files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
    binlen = 0.1
    tslice = [1., 300.]
    
    decoder = train._train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None],
        state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
if test_bc:
    block = 'cart20130521_04'
    files = dict(hdf='/storage/rawdata/hdf/%s.hdf' % block)
    binlen = 0.1
    tslice = [1., 300.]
    
    decoder = _train_KFDecoder_brain_control(cells=None, binlen=0.1, tslice=[None,None],
        state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
