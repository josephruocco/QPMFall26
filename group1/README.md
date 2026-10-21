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
2. Test only AUD/USD against the incoming response. Retain its beta only when the absolute t-statistic is greater than 2; otherwise pass the response through unchanged. Do not test other secondary currencies. Skip AUD if it is already the home-currency prior.
3. Regress the resulting response on SPY; always retain beta and residuals.
4. Regress those residuals on the primary sector residual factor, always retaining beta. Test each remaining sector individually against the current residuals. Keep the factor with the largest absolute t-statistic only if |t| > 2, pass its residuals forward, and repeat with the unselected sectors. Stop when no remaining sector passes. Retained betas are not jointly refitted; there is no count cap.
5. Calculate the final residual covariance matrix on dates shared by every stock. Run centered, unstandardized covariance PCA to obtain all eigenvalues, eigenvectors, and scores.
6. Assemble applied betas, stage and overall fit statistics, both covariance matrices, PCA results, residual histories, and the selection audit in Excel.

Step 4 interprets the primary-sector prior as a mandatory univariate first fit, analogous to the home-currency prior. Its coefficient stays fixed during subsequent one-factor fits. The board's exact-equality case at a t-statistic of +/-2 is ambiguous; this implementation retains only strict absolute t-statistics greater than 2. Empty candidate blocks pass their incoming response through unchanged.

## Source and scope

`data/companies_and_staggered_returns.xlsx` supplies 80 nonoverlapping four-week return dates, already scaled by the class time-weight schedule. Weighted returns start at Excel row 255, and sector residuals at row 339. The company table separates stocks from factors. Duplicate stock selections are modeled once per ticker.

The source contains **43 stocks and 18 factors**: six currency returns, SPY, and eleven sector residual factors. The board sketches 23 factor columns, but the remaining five factors have not been identified or supplied. No factors are fabricated to force that shape. Sector factors are residuals from the upstream SPY regressions, not raw sector ETF returns.

Each stock uses a fixed complete-case sample across every stage. Missing pre-listing prices are never filled. Covariance and PCA use **70 shared dates, 2021-04-26 through 2026-08-10**. Factor covariance uses the identical sample. Sample covariance uses denominator n - 1. Using one common sample preserves positive semidefiniteness; pairwise deletion is not used.

## Outputs and interpretation

The workbook's main views are Summary, Betas, Stage risk, Factor covariance, Residual covariance, and PCA. Supporting sheets contain fit diagnostics, eigenvectors, scores, coefficients, the complete selection log, all stage residual histories, common-sample data, source returns, company assignments, and method notes. CSV outputs provide the same analytical results for downstream use.

Applied beta is zero for excluded factors. Each coefficient row now represents one univariate fit. Its alpha and R-squared belong to that selection step. Total alpha sums one intercept per retained fit. Overall reconstruction R-squared is `1 - sum(final residual^2) / sum((original return - mean(original return))^2)`; it is not adjusted R-squared from a simultaneous multivariate regression.

Cumulative R-squared is reported after home FX, other FX, SPY, the primary sector, and other sectors. Each value is measured against the same original stock-return variance: `1 - sum((stage response - mean(stage response))^2) / sum((original return - mean(original return))^2)`. Centering makes the original baseline zero and leaves skipped stages unchanged. Retained regressions include an intercept, so their residuals are already centered. On the fixed stock sample this also equals `1 - (stage residual std / original std)^2`. Do not add individual stage R-squared values. The final cumulative value equals the overall reconstruction R-squared, up to numerical precision. Betas columns U:Y show the five checkpoints; Stage risk includes the baseline and all checkpoints in Q:V.

Standard deviations use class-weighted four-week return units and are not annualized. Covariances and PCA eigenvalues use the squared units. Eigenvectors are unit-length directions, not variance-scaled loadings. Scores project centered residuals onto those directions; eigenvector signs are arbitrary and are oriented consistently for reproducibility.

Each retained one-factor fit cannot increase residual sample variance relative to its incoming response. The pipeline checks block-level monotonicity, exact return reconstruction to 1e-10, eigenvector orthonormality, and covariance reconstruction from the eigenpairs.

Sequential betas depend on stage order; later fits can reintroduce correlation with earlier factors. OLS on the saved scaled returns differs from conventional weighted least squares. T-statistics are classical OLS statistics. Selection and residual reductions are exploratory and in-sample; no multiple-testing adjustment or out-of-sample validation is applied. Foreign tickers are US-listed USD-priced shares/ADRs; their currency regression measures FX exposure rather than converting local returns into USD.
