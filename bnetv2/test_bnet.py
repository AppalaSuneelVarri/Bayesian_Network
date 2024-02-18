import unittest
import pandas as pd
from io import StringIO
from bnet import BayesianNetwork, VariableElimination

class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        self.training_data = pd.read_csv(StringIO("""0 0 0 0
1 1 1 1
0 1 0 0
1 0 1 0
1 1 0 1
0 0 1 0
1 1 1 1
1 0 0 0
1 1 0 1
0 0 1 0
0 1 1 1
1 0 1 0
1 1 0 1
0 0 0 0
0 1 1 1"""), header=None, sep=' ', dtype=int)
        self.training_data.columns = ['B', 'G', 'C', 'F']
        self.structure = {'G': ['B'], 'F': ['G', 'C']}
        self.model = BayesianNetwork(structure=self.structure)
        self.model.fit(self.training_data)

    def test_task1(self):
        # Check the conditional probability tables
        cpt = self.model.show_cpt()
        self.assertAlmostEqual(round(cpt['B'][()][1], 4), 0.5333, places=4)
        self.assertAlmostEqual(round(cpt['C'][()][1], 4), 0.5333, places=4)
        self.assertAlmostEqual(round(cpt['G'][(0,)][1], 4), 0.4286, places=4)
        self.assertAlmostEqual(round(cpt['G'][(1,)][1], 4), 0.625, places=4)
        self.assertAlmostEqual(round(cpt['F'][(0, 0)][1], 4), 0.5, places=4)
        self.assertAlmostEqual(round(cpt['F'][(0, 1)][1], 4), 0.5, places=4)
        self.assertAlmostEqual(round(cpt['F'][(1, 0)][1], 4), 0.75, places=4)
        self.assertAlmostEqual(round(cpt['F'][(1, 1)][1], 4), 1.0, places=4)

    def test_task2(self):
        # Calculate the joint probability P(B=t, G=f, C=t, F=f)
        infer = VariableElimination(model=self.model)
        result_prob = infer.query(variables={'B': 1, 'G': 0, 'C': 1, 'F': 0}, evidence={})
        self.assertAlmostEqual(result_prob, 0.07754, places=4)

    def test_task3(self):
        # Calculate the conditional probability P(B=t, G=f | F=f)
        infer = VariableElimination(model=self.model)
        result_prob = infer.query(variables={'B': 1, 'G': 0}, evidence={'F': 0})
        self.assertAlmostEqual(result_prob, 0.2980, places=4)

if __name__ == '__main__':
    unittest.main()
