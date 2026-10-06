# Group 1 stock factor regressions

Group members: Joseph Ruocco, Niharika Madana, Hunter Allen, Samson Lam.

Run from the repository root:

```bash
python -m pip install -r group1/requirements.txt
python group1/stock_factor_regressions.py
```

The source is `data/companies_and_staggered_returns.xlsx`. Group 1 uses `stagger_1` only. The sheet supplies 80 nonoverlapping four-week returns, already multiplied by the class time-weight schedule, and sector ETF residuals from the SPY regressions. The Excel row offsets match the existing Group 0 notebook. Duplicate company picks are modeled once per ticker. Stocks, ETFs, and currencies are separated using company metadata.

1. Foreign stocks: regress the stock return on its home-currency return with an intercept; keep its beta and residuals.
2. Foreign stocks: regress those residuals on AUD/USD. Retain the fitted component only when the absolute coefficient t-statistic exceeds 2. Otherwise the applied AUD beta is zero and the response passes through unchanged.
3. US stocks: apply the same conditional AUD regression directly to stock returns.
4. Regress the resulting response on SPY; retain the beta and residuals.
5. Regress those residuals on the stock's primary sector residual factor; always retain the beta and residuals.
6. Forward-select remaining sector residual factors. At each iteration, jointly fit the selected factors and each candidate to the fixed step-5 response. Add the admissible candidate with the largest absolute t-statistic. An addition is admissible only if every selected additional-sector coefficient remains above the cutoff after refitting. Stop when no addition passes.

## Outputs

`results/stock_factor_results.xlsx` contains stage coefficients, a wide factor-exposure table, diagnostics, the candidate-selection log, and the Group 1 final residual table. The same tables are saved as CSV files. Tested coefficients and applied coefficients are separate fields, so rejected AUD estimates remain inspectable. Factor exposures use sector residual returns rather than raw sector ETF returns; an omitted additional sector has zero exposure.

The pipeline uses a common complete-case sample within each stock model. Missing pre-listing prices are never filled. All regressions include an intercept; retained stages subtract both alpha and beta times the factor. Each modeled return is checked against the sum of its retained fitted components and final residual to a tolerance of 1e-10.

## Assumptions

The board leaves the precise stepwise algorithm unspecified; step 6 implements forward selection with joint refits and the stated absolute-t cutoff. The earlier stages follow the board's sequential residual regressions. Sequential betas are order-dependent and differ from a simultaneous multivariate regression. Later regressions can reintroduce correlation with earlier factors because the regressors are not mutually orthogonal.

The saved class weighting is preserved: OLS is applied to already scaled returns with an ordinary intercept. This differs from conventional weighted least squares. T-statistics are classical OLS statistics, and the selection threshold is exploratory; no adjustment for multiple testing or out-of-sample validation is performed. This analysis uses only stagger_1.

Foreign tickers in this workbook are US-listed shares/ADRs with USD prices. Their home-currency regressions measure FX exposure of those USD returns. They do not convert a locally quoted stock return into USD.
