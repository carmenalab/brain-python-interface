from riglib.bmi.extractor import LFPMTMPowerExtractor
import numpy as np

import unittest

class TestMTM(unittest.TestCase):

    extractor = LFPMTMPowerExtractor(None, [0,1], bands=[(80,100),(100,150)], NW=3)
    print(extractor.npk)
    cont_samples = np.random.normal((2,200))
    feats = extractor.extract_features(cont_samples)
    
if __name__ == '__main__':
    unittest.main()


