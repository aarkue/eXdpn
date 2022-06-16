from exdpn.util import import_log, extend_event_log_with_total_elapsed_time

import unittest
import os


class Test_Extend_Log_With_Total_Elapsed_Time(unittest.TestCase):
    def test_simple(self):
        event_log = import_log(os.path.join(
            os.getcwd(), 'tests', 'util', 'example.xes'))

        extend_event_log_with_total_elapsed_time(event_log)

        self.assertEqual(event_log[0][0]["eXdpn::total_elapsed_time"], 0)
        self.assertEqual(event_log[0][3]["eXdpn::total_elapsed_time"], 694800) 


if __name__ == "__main__":
    unittest.main()