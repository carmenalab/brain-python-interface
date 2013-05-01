import numpy as np

class LIFO():
    def __init__(self, ndim, hist_len):
        self.data = np.zeros([ndim, hist_len])

    def __call__(self, x_t):
        self.data[:,1:] = self.data[:,:-1]
        self.data[:,0] = x_t

class LTIFilter():
    def __init__(self, b, a, ndim=1):
        self.b = b
        self.a = a
        self.input_hist = LIFO(ndim, len(b))
        self.output_hist = LIFO(ndim, len(a))

    def __call__(self, x_t):
        self.input_hist(x_t)
        y_t = -np.sum(self.output_hist.data[:,:-1] * self.a[1:], axis=1) +\
            np.sum(self.input_hist.data * self.b, axis=1)
        self.output_hist(y_t)
        return y_t

if __name__ == '__main__':
    from scipy import signal
    bin = 100
    filtCutoff = 2.0
    norm_pass = filtCutoff/ ((1000/bin)/2)
    signal.butter(2, norm_pass, btype='low', analog=0, output='ba')
    b, a = signal.butter(2, norm_pass, btype='low', analog=0, output='ba')

    filter = LTIFilter(b, a, ndim=2)
    x = np.zeros(12)
    x[0] = 1.0
    x = np.vstack([x, x])
    impl_resp = [filter(x[:,t]) for t in range(x.shape[1])]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(impl_resp)
    plt.show()
