"""Selection and reconstruction checks: python -m unittest discover -s group1."""
import unittest
import numpy as np
import pandas as pd
from stock_factor_regressions import backward_select, fit


class BackwardSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(45)
        basis = rng.normal(size=(200, 5))
        basis -= basis.mean(axis=0)
        self.q, _ = np.linalg.qr(basis)
        self.x = pd.DataFrame(self.q[:, :3], columns=['negative', 'positive', 'noise'])

    def test_preserves_negative_signal_and_removes_noise(self):
        y = pd.Series(2 - 4*self.q[:, 0] + 3*self.q[:, 1] + 0.1*self.q[:, 3])
        model, selected, log = backward_select(y, self.x)
        self.assertEqual(selected, ['negative', 'positive'])
        self.assertEqual([r['factor'] for r in log if r['removed']], ['noise'])
        self.assertTrue((model.tvalues[selected].abs() > 2).all())
        np.testing.assert_allclose(model.fittedvalues + model.resid, y)
        np.testing.assert_allclose(self.x[selected].T @ model.resid, 0, atol=1e-12)

    def test_all_factors_can_be_removed(self):
        y = pd.Series(1 + self.q[:, 4])
        model, selected, log = backward_select(y, self.x)
        self.assertIsNone(model)
        self.assertEqual(selected, [])
        self.assertEqual(sum(r['removed'] for r in log), 3)

    def test_refits_fixed_response_at_each_iteration(self):
        y = pd.Series(1 - 2*self.q[:, 0] + 0.03*self.q[:, 1] + 0.02*self.q[:, 2] + self.q[:, 3])
        _, _, log = backward_select(y, self.x)
        active = list(self.x.columns)
        for iteration in sorted({r['iteration'] for r in log}):
            rows = [r for r in log if r['iteration'] == iteration]
            expected = fit(y, self.x[active])
            for row in rows:
                self.assertAlmostEqual(row['t_stat'], expected.tvalues[row['factor']])
                if row['removed']:
                    self.assertEqual(row['factor'], expected.tvalues[active].abs().idxmin())
            active = [f for f in active if not any(r['factor'] == f and r['removed'] for r in rows)]

    def test_rejects_collinear_design(self):
        x = self.x.copy()
        x['duplicate'] = x['negative']
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            backward_select(pd.Series(self.q[:, 4]), x)


if __name__ == '__main__':
    unittest.main()
