import unittest
import os
import time

from ..riglib.mp_proxy import RPCProcess


class MockTask(object):
    def __init__(self):
        print("MockTask.__init__", self)
        self.run_state = True
        self.call_count = 0
        self.remote_value = 3

    def simple_fn(self, a, b=2):
        return a * b

    def incr_call_count(self):
        # self.call_count += 1
        self.call_count = self.call_count + 2
        # global call_count
        # call_count += 1
        print("call count, self=", self, self.call_count)
        return self.call_count

    def __setattr__(self, attr, value):
        print("MockTask.__setattr__", attr, value)
        super().__setattr__(attr, value)

    def is_running(self):
        # print("mocktask.runstate", self.run_state)
        return self.run_state

class MockRPCTask(RPCProcess):
    def target_constr(self):
        self.target = MockTask()

    def check_run_condition(self):
        return self.target.run_state

    def end_task(self):
        self.target_proxy.set('run_state', False)

    @property
    def remote_class(self):
        return MockTask

    def loop_task(self):
        if hasattr(self.target, 'end_task_flag'):
            print(self.target.run_state)


class TestMpProxy(unittest.TestCase):
    def test_access_of_attributes_and_stateless_functions(self):
        log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        proc = MockRPCTask(log_filename=log_filename)
        task_proxy, data_proxy = proc.start()

        # @test access to remote value through the proxy
        x = task_proxy.remote_value
        self.assertEqual(x, 3)

        # @test access to remote value through the getattr redirect
        x = proc.remote_value
        self.assertEqual(x, 3)             

        # @test remote function call
        prod = task_proxy.simple_fn(2, b=3)
        self.assertEqual(prod, 6)        

        self.assertEqual(task_proxy.simple_fn(2), 4)                

        proc.stop()

    def test_modifying_attr(self):
        log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        proc = MockRPCTask(log_filename=log_filename)
        task_proxy, data_proxy = proc.start()

        task_proxy.set('remote_value', 4)

        self.assertEqual(task_proxy.remote_value, 4)
        self.assertTrue('remote_value' not in task_proxy.__dict__)
        proc.stop()

    def test_end_task(self):
        log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        proc = MockRPCTask(log_filename=log_filename)
        task_proxy, data_proxy = proc.start()
        time.sleep(1)

        proc.end_task()
        proc.join()


    def test_in_remote_process(self):
        log_filename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        proc = MockRPCTask(log_filename=log_filename)
        task_proxy, data_proxy = proc.start()

        # # @test remote function call through redirect
        for k in range(2):
            fn = task_proxy.incr_call_count
            print("Function object", fn)
            print(fn())
            # print("reported call count", task_proxy.incr_call_count())
            # print("reported call count", proc.incr_call_count())
            print("proc.target", proc.target)
            time.sleep(1)
            
        # time.sleep(1)
        # print(proc.incr_call_count())
        # # time.sleep(1)
        # print(proc.incr_call_count())
        # print(proc.incr_call_count())
        # print(proc.incr_call_count())
        # print(proc.incr_call_count())

        # print(proc.__dict__)
        self.assertFalse('call_count' in task_proxy.__dict__)

        proc.stop()

if __name__ == '__main__':
    unittest.main()
