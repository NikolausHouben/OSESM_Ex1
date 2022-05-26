from funcs import run_bandits, report_bandits
import unittest

class TestFuncs(unittest.TestCase):
    def test_run_bandits(self):
        self.assertRaises(ValueError, run_bandits, 10.1, 1000, 2, 0.3)

    def test_report_bandits(self):

        dictio = run_bandits(10, 1000, 2, 0.3)
        best_epsilon = report_bandits(10, dictio)
        self.assertGreater(best_epsilon, 0)


if __name__ == "__main__":
    unittest.main()
