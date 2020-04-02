import unittest
import tables
import os
import numpy as np

from hdfwriter import HDFWriter


class TestHDFWriter(unittest.TestCase):
    test_output_fname = "test.hdf"
    def setUp(self):
        self.wr = wr = HDFWriter(self.test_output_fname)
        self.table1_dtype = np.dtype([("stuff", np.float64)])
        self.table2_dtype = np.dtype([("stuff2", np.float64), ("stuff3", np.uint8)])
        wr.register("table1", self.table1_dtype, include_msgs=True)
        wr.register("table2", self.table2_dtype, include_msgs=False)

        # send some data
        wr.send("table1", np.zeros(3, dtype=self.table1_dtype))
        wr.send("table1", np.ones(1, dtype=self.table1_dtype))
        wr.send("table2", np.ones(1, dtype=self.table2_dtype))
        wr.sendMsg("message!")
        wr.close()

    def test_h5_file_created(self):
        h5 = tables.open_file(self.test_output_fname)
        self.assertTrue(hasattr(h5, "root"))
        h5.close()

    def test_tables_exist(self):
        h5 = tables.open_file(self.test_output_fname)
        self.assertTrue(hasattr(h5.root, "table1"))
        self.assertTrue(hasattr(h5.root, "table1_msgs"))
        self.assertTrue(hasattr(h5.root, "table2"))
        self.assertFalse(hasattr(h5.root, "table2_msgs"))

        self.assertEqual(len(h5.root.table1), 4) # NOTE this only works after a bugfix in HDFWriter
        self.assertEqual(len(h5.root.table2), 1)
        self.assertEqual(len(h5.root.table1_msgs), 1)

        self.assertEqual(h5.root.table1_msgs[0]['msg'].decode("utf-8"), "message!")
        self.assertEqual(h5.root.table1_msgs[0]['time'], 4)

        self.assertTrue(np.all(h5.root.table2[:]['stuff2'] == 1))
        h5.close()


    def tearDown(self):
        if os.path.exists(self.test_output_fname):
            os.remove(self.test_output_fname)

if __name__ == '__main__':
    unittest.main()

