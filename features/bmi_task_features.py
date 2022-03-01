'''
BMI task features
'''
import time
import numpy as np
from riglib.experiment import traits, experiment

###### CONSTANTS
sec_per_min = 60

########################################################################################################
# Decoder/BMISystem add-ons
########################################################################################################
class NormFiringRates(traits.HasTraits):
    ''' Docstring '''
    
    norm_time = traits.Float(120., desc="Number of seconds to use for mean and SD estimate")

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raise NotImplementedError("This feature is extremely depricated and probably does not work properly anymore.")
        super(NormFiringRates, self).__init__(*args, **kwargs)
        import time
        self.starttime = time.time()
        self.elapsedtime=0
        self.count=0
        self.mFR = None
        self.mFR2 = None
        self.sdFR = None
        self.updated=False

    def update_fr_vals(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raise NotImplementedError("decoder.bin_spikes no longer exists, use the feature extractor instead")
        if self.elapsedtime>1.:
            bin = self.decoder.bin_spikes(self.neurondata.get(all=True).copy())
            self.count +=1
            if self.count == 1:
                sz = len(bin)
                self.mFR=np.zeros([sz])
                self.mFR2=np.zeros([sz])
            delta = bin - self.mFR
            self.mFR = self.mFR + delta/self.count
            self.mFR2 = self.mFR2 + delta*(bin - self.mFR)

    def update_cursor(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.elapsedtime = time.time()-self.starttime
        if self.elapsedtime<self.norm_time:
            self.update_fr_vals()
        elif not self.updated:
            self.sdFR = np.sqrt(self.mFR2/(self.count-1))
            self.decoder.init_zscore(self.mFR,self.sdFR)
            self.hdf.sendMsg("baseline_norm")
            self.updated=True
            print("Updated session mean and SD.")
            self.hdf.sendAttr("task", "session_mFR", self.mFR)
            self.hdf.sendAttr("task", "session_sdFR", self.sdFR)

        super(NormFiringRates, self).update_cursor()

class LinearlyDecreasingAttribute(traits.HasTraits):
    ''' 
    Generic feature which linearly decreases the value of attributes used by the task
    '''
    attrs = []
    attr_flags = dict()
    def __init__(self, *args, **kwargs):
        '''
        Constructor for LinearlyDecreasingAttribute

        Parameters
        ----------
        *args, **kwargs: None necessary, these are for multiple inheritance compatibility

        Returns
        -------
        LinearlyDecreasingAttribute instance
        '''
        assert isinstance(self, experiment.Experiment)
        super(LinearlyDecreasingAttribute, self).__init__(*args, **kwargs)

    def init(self):
        '''
        Secondary init function after the object has been created and all the super __init__ functions have run.
        Initialize the "current" value of all of the attributes and register the attributes with the HDF file
        '''

        if hasattr(self, 'add_dtype'):
            for attr in self.attrs:
                if attr not in self.dtype:
                    self.add_dtype(attr, 'f8', (1,))
        
        super(LinearlyDecreasingAttribute, self).init()
        for attr in self.attrs:
            self.attr_start, self.attr_min = getattr(self, attr)
            setattr(self, 'current_%s' % attr, self.attr_start)
            self.attr_flags[attr] = True


    def _linear_change(self, start_val, end_val, decay_time):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if start_val == end_val:
            return end_val
        else:
            elapsed_time = self.get_time() - self.task_start_time
            temp = start_val - elapsed_time/decay_time*(start_val-end_val)
            if start_val > end_val:
                return max(temp, end_val)
            elif start_val < end_val:
                return min(temp, end_val)

    def update_level(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        for attr in self.attrs:
            decay_time = float(getattr(self, '%s_time' % attr))
            attr_start, attr_min = getattr(self, attr)
            current_level = self._linear_change(attr_start, attr_min, decay_time)
            setattr(self, 'current_%s' % attr, current_level)
            flag = self.attr_flags[attr]
            if flag and getattr(self, 'current_%s' % attr) == attr_min:
                print("%s at final value after %d successful trials" % (attr, self.calc_state_occurrences('reward')))
                self.attr_flags[attr] = False

            if self.cycle_count % (1./self.update_rate * sec_per_min) == 0 and self.attr_flags[attr]:
                print("%s: " % attr, getattr(self, 'current_%s' % attr))

    def _cycle(self):
        '''
        Update and save the current attribute value before calling the next _cycle in the MRO
        '''
        self.update_level()
        if hasattr(self, 'task_data'):
            for attr in self.attrs:
                self.task_data[attr] = getattr(self, 'current_%s' % attr)

        super(LinearlyDecreasingAttribute, self)._cycle()

class LinearlyDecreasingAssist(LinearlyDecreasingAttribute):
    '''
    Specific case of LinearlyDecreasingAttribute for a linearly decreasing assist parameter
    '''
    assist_level = traits.Tuple((0.0, 0.0), desc="Level of assist to apply to BMI output")
    assist_level_time = traits.Float(600, desc="Number of seconds to go from initial to minimum assist level")

    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingAssist, self).__init__(*args, **kwargs)
        if 'assist_level' not in self.attrs:
            self.attrs.append('assist_level')

class LinearlyDecreasingXYAssist(LinearlyDecreasingAttribute):
    ''' 
    linearly decreasing XY assist -- for ArmAssist
    '''
    aa_assist_level = traits.Tuple((0.0, 0.0), desc='level of assist to apply to XY output')
    aa_assist_level_time = traits.Float(600, desc="Number of seconds to go from initial to minimum assist level")
    
    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingXYAssist, self).__init__(*args, **kwargs)
        if 'aa_assist_level' not in self.attrs:
            self.attrs.append('aa_assist_level')

class LinearlyDecreasingAngAssist(LinearlyDecreasingAttribute):
    ''' 
    linearly decreasing angular assist -- for psi and ReHand
    '''
    rh_assist_level = traits.Tuple((0.0, 0.0), desc='level of assist to apply to ang output')
    rh_assist_level_time = traits.Float(600, desc="Number of seconds to go from initial to minimum assist level")
    
    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingAngAssist, self).__init__(*args, **kwargs)
        if 'rh_assist_level' not in self.attrs:
            self.attrs.append('rh_assist_level')

class LinearlyDecreasingHalfLife(LinearlyDecreasingAttribute):
    '''
    Specific case of LinearlyDecreasingAttribute for a linearly decreasing CLDA half-life
    '''
    half_life = traits.Tuple((450., 450.), desc="Initial and final half life for CLDA")
    half_life_time = traits.Float(600, desc="Number of seconds to go from initial to final half life")
    
    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingHalfLife, self).__init__(*args, **kwargs)
        if 'half_life' not in self.attrs:
            self.attrs.append('half_life')    

class LinearlyDecreasingReachAngle(LinearlyDecreasingAssist):
    '''
    For the reach direction task, decrease the maximum reach angle linearly
    '''
    reach_angle_time = traits.Float(600, desc="Number of seconds to go from initial to final reach angle")

    def __init__(self, *args, **kwargs):
        super(LinearlyDecreasingHalfLife, self).__init__(*args, **kwargs)
        if 'half_life' not in self.attrs:
            self.attrs.append('half_life')    