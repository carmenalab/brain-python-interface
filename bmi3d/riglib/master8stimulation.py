'''
Code for the triggering stimulation from the Master-8 voltage-programmable stimulator
'''
import comedi

###### CONSTANTS
sec_per_min = 60

"""
Define one class to send TTL pulse for one cycle of stimulation (StimulationPulse).  Define second class to implement this
at a fixed rate.

Left off at line 79.  Need to figure out how we re-trigger starting the stimulus pulse train again.
"""



class TTLStimulation(object):
    '''During the stimulation phase, send a timed TTL pulse to the Master-8 stimulator'''
    hold_time = float(1)
    stimulation_pulse_length = float(0.2*1e6)
    stimulation_frequency = float(3)
    
    status = dict(
        pulse = dict(pulse_end="interpulse_period", stop=None),
        interpulse_period = dict(another_pulse="pulse", pulse_train_end="pulse_off", stop=None),
        pulse_off= dict(stop=None)
    )

    com = comedi.comedi_open('/dev/comedi0')
    pulse_count = 0  #initializing number of pulses that have occured

    def __init__(self, *args, **kwargs):
        super(TTLStimulation, self).__init__(*args, **kwargs)
        number_of_pulses = int(self.hold_time*self.stimulation_frequency)   # total pulses during a stimulation pulse train, assumes hold_time is in s

    def init(self):
        super(TTLStimulation, self).init()

    #### TEST FUNCTIONS ####

    def _test_pulse_end(self, ts):
        #return true if time has been longer than the specified pulse duration
        pulse_length = self.stimulation_pulse_length*1e-6    # assumes stimulation_pulse_length is in us
        return ts>=pulse_length

    def _test_another_pulse(self,ts):
        interpulse_time = (1/self.stimulationfrequency) - self.stimulation_pulse_length*1e-6    # period minus the duration of a pulse
        return ts>=interpulse_time

    def _test_pulse_train_end(self,ts):
        return (self.pulse_count > number_of_pulses)     # end train if number of pulses is completed or if the animal ends holding early


    #### STATE FUNCTIONS ####

    def _start_pulse(self):
        '''
        At the start of the stimulation state, send TTL pulse
        '''

        #super(TTLStimulation, self)._start_pulse()
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.pulse_count = self.pulse_count + 1
        #self.stimulation_start = self.get_time() - self.start_time

    def _end_pulse(self):
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

    def _start_interpulse_period(self):
        super(TTLStimulation, self)._start_interpulse_period()
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

    def _end_interpulse_period(self):
        super(TTLStimulation, self)._end_interpulse_period()
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

