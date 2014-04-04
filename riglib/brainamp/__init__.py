class EMG(object):
    update_freq = 2000.  # TODO -- double check

    dtype = np.dtype('float')

    def __init__(self, channels):
        self.conn = rda.Connection()
        self.conn.connect()
        # self.conn.select_channels(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()

        return (d.chan, np.array([d.uV_value], dtype='float'))
