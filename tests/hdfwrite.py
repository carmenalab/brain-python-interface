import time
import numpy as np
from riglib import hdfwriter, sink

def test(filename, shape=(8,4), length=2**28):
    dtype = np.dtype((np.float, shape))
#    h5 = sink.sinks.start(hdfwriter.HDFWriter, filename=filename)
    h5 = hdfwriter.HDFWriter(filename)
    h5.register("test", dtype)

    times = np.zeros(length)
    times[0] = time.time()
    for i in xrange(1, int(1e7)):
        h5.send("test", np.random.randn(*shape))
        times[i] = time.time()
        print i

    np.save("/tmp/times.npy", times)

if __name__ == "__main__":
#    import tempfile
#    tf = tempfile.NamedTemporaryFile()
    test("/storage/test.hdf")
