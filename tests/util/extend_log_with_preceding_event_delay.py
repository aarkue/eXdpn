from exdpn.util import import_log, extend_event_log_with_preceding_event_delay

import unittest
import os


class Test_Extend_Log_With_Preceding_Event_Delay(unittest.TestCase):
    def test_simple(self):
        event_log = import_log(os.path.join(
            os.getcwd(), 'tests', 'util', 'example.xes'))

        extend_event_log_with_preceding_event_delay(event_log)
        
        self.assertEqual(event_log[1][1]["eXdpn::preceding_event_delay"], 3600)

if __name__ == "__main__":
    unittest.main()