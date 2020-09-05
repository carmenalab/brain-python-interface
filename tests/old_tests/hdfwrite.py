import time
import numpy as np
from riglib import hdfwriter, sink

def test(filename, shape=(8,4), length=2**28):
    dtype = np.dtype((np.float, shape))
    h5 = sink.sinks.start(hdfwriter.HDFWriter, filename=filename)
    h5.register("test", dtype)

    times = np.zeros(length)
    times[0] = time.time()
    data = np.random.randn(*shape)
    for i in range(1, int(length)):
        h5.send("test", data)
        times[i] = time.time()
    print("I'm done!")
    h5.stop()
    h5.join()
    np.save("/tmp/times.npy", times)

if __name__ == "__main__":
#    import tempfile
#    tf = tempfile.NamedTemporaryFile()
    test("/tmp/test.hdf", length=1e6)
