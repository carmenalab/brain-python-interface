class EMG(object):
    update_freq = 1000.  # TODO -- double check

    dtype = np.dtype('float')

    def __init__(self, channels):
        self.conn = cerelink.Connection()
        self.conn.connect()
        self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_continuous_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()

        # TODO - need to convert samples to mV first?
        return (d.chan, d.samples)