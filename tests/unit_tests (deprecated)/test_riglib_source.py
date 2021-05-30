import numpy as np
import unittest
from reqlib import swreq
from requirements import *

from ..riglib import source
import time

class MockDataSourceSystem(source.DataSourceSystem):
    update_freq = 2000
    dtype = np.dtype([("value", np.float)])
    attr_to_test_access = 43
    
    state = 0
    delay_for_get = 0.010

    def get(self):
        time.sleep(self.delay_for_get)
        self.state += 1
        if self.state >= 255:
            self.state = 0
        return np.array([self.state], dtype=self.dtype)

    def procedure(self):
        return 42

class MockDataSourceSystem2(MockDataSourceSystem):
    """ Same as original, but with simple built-in dtype """
    dtype = np.dtype("float")

class MockDataSourceSystem3(MockDataSourceSystem):
    delay_for_get = 0.0


class TestDataSourceSystem(unittest.TestCase):
    @swreq(req_source)
    def test_basic_data_source_get(self):
        mock_dss = MockDataSourceSystem()
        mock_data = mock_dss.get()
        self.assertEqual(len(mock_data), 1)

    def test_source_polling(self):
        src = source.DataSource(MockDataSourceSystem, send_data_to_sink_manager=False)
        src.start()

        data_all = []
        for k in range(60):
            data = src.get()
            data_all.append(data)
            time.sleep(0.100)

        src.stop()
        del src

        for data in data_all:
            if len(data) > 0:
                self.assertEqual((data[0]["value"] + len(data) - 1) % 255, data[-1]["value"])

    def test_source_polling_fast(self):
        src = source.DataSource(MockDataSourceSystem3, send_data_to_sink_manager=False)
        src.start()

        data_all = []
        for k in range(60):
            data = src.get()
            data_all.append(data)
            time.sleep(0.100)

        src.stop()
        del src

        for data in data_all:
            if len(data) > 0:
                self.assertEqual((data[0]["value"] + len(data) - 1) % 255, data[-1]["value"])

    def test_source_polling2(self):
        src = source.DataSource(MockDataSourceSystem2, send_data_to_sink_manager=False)
        src.start()

        data_all = []
        for k in range(60):
            data = src.get()
            data_all.append(data)
            time.sleep(0.100)

        src.stop()
        del src

        for data in data_all:
            if len(data) > 0:
                self.assertEqual((data[0] + len(data) - 1) % 255, data[-1])

    def test_source_get_all(self):
        """source.get(all=True) should produce a growing output"""
        src = source.DataSource(MockDataSourceSystem, send_data_to_sink_manager=False)
        src.start()

        prev_len = -1
        data_all = []
        for k in range(60):
            data = src.get(all=True)
            data_all.append(data)
            time.sleep(0.100)
            self.assertTrue(len(data) >= prev_len)
            prev_len = len(data)

        src.stop()
        del src            

    def test_rpc(self):
        src = source.DataSource(MockDataSourceSystem, send_data_to_sink_manager=False, log_filename='test_riglib_source.log')
        src.start()

        self.assertEqual(src.procedure(), 42)
        self.assertEqual(src.attr_to_test_access, 43)

if __name__ == '__main__':
    unittest.main()

