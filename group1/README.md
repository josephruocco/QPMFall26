# Group 1 stock factor analysis

Group members: Joseph Ruocco, Niharika Madana, Hunter Allen, Samson Lam.

The analysis uses **stagger_1** and the corrected upstream data. Open `results/group1_stock_factor_results.xlsx` for the complete formatted workbook, or run `Stock_Factor_Regressions.ipynb` to walk through the calculations and plots.

```bash
python -m pip install -r group1/requirements.txt
python group1/stock_factor_regressions.py
python -m unittest discover -s group1
```

The script and notebook refresh the CSV outputs. The formatted Excel workbook is a snapshot of the committed results and requires a separate rebuild after rerunning the analysis. Its ratios and covariance matrices use Excel formulas; regressions, variable selection, and PCA are computed in Python and do not refit when workbook inputs are edited.

## Classroom method

1. Foreign stocks: regress the weighted stock return on its home currency with an intercept. Always retain beta and pass residuals forward. US stocks skip this step.
2. Jointly regress the incoming response on all remaining currencies. Remove the coefficient with the smallest absolute t-statistic when its absolute t-statistic is at most 2, then refit against the same incoming response. Continue until all remaining coefficients have absolute t-statistics greater than 2 or all factors are removed. US stocks start with all six currencies.
3. Regress the resulting response on SPY; always retain beta and residuals.
4. Regress those residuals on the primary sector residual factor, always retaining beta. Apply the same backward-elimination procedure to all ten remaining sector residual factors.
5. Calculate the final residual covariance matrix on dates shared by every stock. Run centered, unstandardized covariance PCA to obtain all eigenvalues, eigenvectors, and scores.
6. Assemble applied betas, stage and overall fit statistics, both covariance matrices, PCA results, residual histories, and the elimination audit in Excel.

Step 4 interprets the primary-sector prior as a mandatory univariate first fit, analogous to the home-currency prior. It is not forced into the later joint fit. The board's exact-equality case at a t-statistic of +/-2 is ambiguous; this implementation retains only strict absolute t-statistics greater than 2. Empty candidate blocks pass their incoming response through unchanged.

## Source and scope

`data/companies_and_staggered_returns.xlsx` supplies 80 nonoverlapping four-week return dates, already scaled by the class time-weight schedule. Weighted returns start at Excel row 255, and sector residuals at row 339. The company table separates stocks from factors. Duplicate stock selections are modeled once per ticker.

The source contains **43 stocks and 18 factors**: six currency returns, SPY, and eleven sector residual factors. The board sketches 23 factor columns, but the remaining five factors have not been identified or supplied. No factors are fabricated to force that shape. Sector factors are residuals from the upstream SPY regressions, not raw sector ETF returns.

Each stock uses a fixed complete-case sample across every stage. Missing pre-listing prices are never filled. Covariance and PCA use **70 shared dates, 2021-04-26 through 2026-08-10**. Factor covariance uses the identical sample. Sample covariance uses denominator n - 1. Using one common sample preserves positive semidefiniteness; pairwise deletion is not used.

## Outputs and interpretation

The workbook's main views are Summary, Betas, Stage risk, Factor covariance, Residual covariance, and PCA. Supporting sheets contain fit diagnostics, eigenvectors, scores, coefficients, the complete selection log, all stage residual histories, common-sample data, source returns, company assignments, and method notes. CSV outputs provide the same analytical results for downstream use.

Applied beta is zero for excluded factors. Repeated alpha and R-squared entries within a joint stage fit refer to that same fit and must not be summed once per factor. Total alpha sums one intercept per retained fit. Overall reconstruction R-squared is `1 - sum(final residual^2) / sum((original return - mean(original return))^2)`; it is not adjusted R-squared from a simultaneous multivariate regression.

Standard deviations use class-weighted four-week return units and are not annualized. Covariances and PCA eigenvalues use the squared units. Eigenvectors are unit-length directions, not variance-scaled loadings. Scores project centered residuals onto those directions; eigenvector signs are arbitrary and are oriented consistently for reproducibility.

Retained block fits cannot increase residual sample variance relative to their incoming response. Removing regressors *within* backward elimination can increase variance relative to the larger candidate fit. The pipeline checks block-level monotonicity, exact return reconstruction to 1e-10, eigenvector orthonormality, and covariance reconstruction from the eigenpairs.

Sequential betas depend on stage order; later fits can reintroduce correlation with earlier factors. OLS on the saved scaled returns differs from conventional weighted least squares. T-statistics are classical OLS statistics. Selection and residual reductions are exploratory and in-sample; no multiple-testing adjustment or out-of-sample validation is applied. Foreign tickers are US-listed USD-priced shares/ADRs; their currency regression measures FX exposure rather than converting local returns into USD.
