"""Selection and reconstruction checks: python -m unittest discover -s group1."""
import unittest
import numpy as np
import pandas as pd
from stock_factor_regressions import (backward_select, fit, cumulative_r_squared,
                                     forward_residual_select, secondary_currencies)


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

    def test_forward_selection_updates_residuals_and_keeps_negative_beta(self):
        y = pd.Series(2 - 4*self.q[:, 0] + 3*self.q[:, 1] + 0.1*self.q[:, 3])
        retained, log = forward_residual_select(y, self.x)
        self.assertEqual([f for f, _ in retained], ['negative', 'positive'])
        response = y.copy()
        contribution = pd.Series(0., index=y.index)
        for iteration, (factor, model) in enumerate(retained, 1):
            np.testing.assert_allclose(model.model.endog, response)
            candidates = [r for r in log if r['iteration'] == iteration]
            accepted = [r for r in candidates if r['retained']]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]['factor'], factor)
            self.assertEqual(abs(accepted[0]['t_stat']), max(abs(r['t_stat']) for r in candidates))
            self.assertGreater(abs(accepted[0]['t_stat']), 2)
            self.assertLessEqual(model.resid.var(), response.var())
            contribution += model.fittedvalues
            response = model.resid
        np.testing.assert_allclose(contribution + response, y, atol=1e-12)
        self.assertFalse(log[-1]['retained'])

    def test_forward_selection_rejects_noise_and_exact_threshold(self):
        y = pd.Series(1 + self.q[:, 4])
        retained, log = forward_residual_select(y, self.x)
        self.assertEqual(retained, [])
        self.assertTrue(all(not r['retained'] for r in log))
        y = pd.Series(self.q[:, 0] + self.q[:, 3])
        x = self.x[['negative']]
        threshold = abs(fit(y, x).tvalues['negative'])
        self.assertEqual(forward_residual_select(y, x, threshold)[0], [])
        self.assertEqual(forward_residual_select(y, x.iloc[:, :0]), ([], []))

    def test_secondary_currency_is_only_aud(self):
        for home in [None, 'MXN/USD', 'SGD/USD', 'CAD/USD', 'EUR/USD', 'DKK/USD']:
            self.assertEqual(secondary_currencies(home), ['AUD/USD'])
        self.assertEqual(secondary_currencies('AUD/USD'), [])

    def test_cumulative_r_squared_uses_original_baseline(self):
        original = pd.Series(2 + 4*self.q[:, 0] + 3*self.q[:, 1] + self.q[:, 3])
        first = fit(original, self.x[['negative']])
        second = fit(first.resid, self.x[['positive']])
        self.assertEqual(cumulative_r_squared(original, original), 0)
        self.assertAlmostEqual(cumulative_r_squared(original, first.resid), first.rsquared)
        cumulative = cumulative_r_squared(original, second.resid)
        self.assertAlmostEqual(cumulative, 1 - (1-first.rsquared)*(1-second.rsquared))
        self.assertNotAlmostEqual(cumulative, first.rsquared + second.rsquared)
        self.assertAlmostEqual(cumulative_r_squared(original, original+100), 0)
        with self.assertRaisesRegex(ValueError, 'same observations'):
            cumulative_r_squared(original, second.resid.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'constant'):
            cumulative_r_squared(pd.Series([1.,1.,1.]), pd.Series([0.,0.,0.]))


if __name__ == '__main__':
    unittest.main()
