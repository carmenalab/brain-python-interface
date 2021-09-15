from riglib.experiment import traits
import aopy
import numpy as np
import matplotlib.pyplot as plt
import os

class SpatialPlotRMS(traits.HasTraits):
    '''
    Plot RMS of each electrode in spatial coordinates in a separate window. To be used with the CorticalData 
    feature enabled and self.cortical_channels already set.
    '''

    spatial_plot_fps = traits.Float(10, desc="At what frequency to update the spatial plot")
    spatial_plot_window = traits.Float(2, desc="Length of window to average RMS power")
    samplerate = 1000.

    # Channel mapping -> to be replaced by drmap-ui
    channel_map = os.path.join("/home/pagaiisland/drmap", "210910_ecog_signal_path.xlsx")
    position_map = os.path.join("/home/pagaiisland/drmap", "244ch_viventi_ecog_elec_to_pos.xlsx")

    def init(self):
        self.decoder = object() # this is really dumb, but i don't feel like changing a whole bunch of other things, so...
        self.decoder.extractor_cls = object()
        self.decoder.extractor_cls.feature_type = 'lfp'
        super().init()

        # Init channel mapping
        n_channels = len(self.cortical_channels)
        self.channel_idx = list(range(n_channels))
        channels = [ch+1 for ch in self.channel_idx]
        # self.x_pos, self.y_pos = aopy.preproc.drmap_ch_to_pos(channels, self.position_map)
        self.x_pos = list(range(240))
        self.y_pos = list(range(240))

        # Init figure
        fig, ax = plt.subplots(figsize=(8,8))
        zerodata = np.zeros((len(self.cortical_channels),))
        data_map = aopy.visualization.get_data_map(zerodata, self.x_pos, self.y_pos)
        self.spatial_plot = aopy.visualization.plot_spatial_map(data_map, self.x_pos, self.y_pos, ax)

    def _cycle(self):
        count = int(self.fps/self.spatial_plot_fps)
        if self.cycle_count % count == 0:
            
            # Calculate RMS
            n_samples = self.spatial_plot_window * self.samplerate
            neural_data = self.neurondata.get(n_samples, self.channel_idx)
            rms = aopy.analysis.calc_rms(neural_data)

            # Apply channel mapping
            # rms_elec = aopy.preproc.drmap_data_to_elec(rms, self.channel_map)
            rms_elec = rms

            # Update the image
            data_map = aopy.visualization.get_data_map(rms_elec, self.x_pos, self.y_pos)
            self.spatial_plot.set_data(data_map)

        super()._cycle()