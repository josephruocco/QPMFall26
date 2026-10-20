"""Group 1 block regressions, residual covariance, and covariance-based PCA."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SECTORS = {
    'Communication Services': 'XLC', 'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Financials': 'XLF',
    'Health Care': 'XLV', 'Industrials': 'XLI', 'Materials': 'XLB',
    'Real Estate': 'XLRE', 'Technology': 'XLK', 'Utilities': 'XLU',
}
STAGES = ['Original', 'Home FX', 'Other FX', 'SPY', 'Primary sector', 'Other sectors']


def fit(y, x):
    design = sm.add_constant(x, has_constant='add')
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError('Rank-deficient regression design')
    if len(y) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom')
    return sm.OLS(y, design).fit()


def backward_select(y, x, threshold=2.0):
    """Drop the smallest absolute t-statistic, refitting the fixed response."""
    selected, log = sorted(x.columns), []
    iteration = 0
    while selected:
        iteration += 1
        model = fit(y, x[selected])
        t = model.tvalues[selected]
        if not np.isfinite(t).all():
            raise ValueError('Nonfinite t-statistic during backward elimination')
        weakest = t.abs().idxmin()
        # Resolve the board's ambiguous equality case using strict |t| > 2.
        removed = weakest if abs(t[weakest]) <= threshold else None
        for factor in selected:
            log.append(dict(iteration=iteration, factor=factor,
                            beta=float(model.params[factor]), t_stat=float(t[factor]),
                            removed=factor == removed, final_fit=removed is None,
                            r_squared=float(model.rsquared)))
        if removed is None:
            return model, selected, log
        selected.remove(removed)
    return None, [], log


def run(output_dir=None):
    source = ROOT / 'data/companies_and_staggered_returns.xlsx'
    output = Path(output_dir) if output_dir is not None else ROOT / 'group1/results'
    stagger = 'stagger_1'
    companies = pd.read_excel(source, sheet_name='Companies')
    companies = companies.dropna(subset=['sector', 'currency']).drop_duplicates('ticker')
    returns = pd.read_excel(source, sheet_name=stagger, skiprows=254, nrows=80)
    sectors = pd.read_excel(source, sheet_name=stagger, skiprows=338, nrows=80)
    sectors = sectors.loc[:, ~sectors.columns.astype(str).str.startswith('Unnamed:')]
    for frame in [returns, sectors]:
        frame['date'] = pd.to_datetime(frame['date'])
        frame.set_index('date', inplace=True)
    currencies = [c for c in returns if c.endswith('/USD')]
    factor_names = currencies + ['SPY'] + list(SECTORS.values())
    factors = returns[currencies + ['SPY']].join(
        sectors.rename(columns=lambda c: c.removesuffix('_residual')))[factor_names]
    records, diagnostics, selection, progress, exposures = [], [], [], [], []
    residuals = pd.DataFrame(index=returns.index)
    stage_returns = {stage: pd.DataFrame(index=returns.index) for stage in STAGES}
    for company in companies.itertuples(index=False):
        ticker, primary = company.ticker, SECTORS[company.sector]
        home = f'{company.currency}/USD' if company.currency != 'USD' else None
        data = returns[[ticker]].join(factors).replace([np.inf, -np.inf], np.nan).dropna()
        original = data[ticker].copy()
        y = original.copy()
        contribution = pd.Series(0.0, index=y.index)
        alpha_total = 0.0
        exposure = dict(stagger=stagger, ticker=ticker,
                        **{f'beta_{f}': 0.0 for f in factor_names})

        def checkpoint(stage):
            stage_returns[stage][ticker] = y
            progress.append(dict(stagger=stagger, ticker=ticker, stage=stage,
                                 observations=len(y), residual_std=float(y.std(ddof=1))))

        def apply(model, selected, stage):
            nonlocal y, contribution, alpha_total
            if model is None:
                return
            contribution += model.fittedvalues
            y = model.resid.copy()
            alpha_total += float(model.params['const'])
            for factor in selected:
                exposure[f'beta_{factor}'] = float(model.params[factor])
                records.append(dict(stagger=stagger, ticker=ticker, stage=stage,
                    factor=factor, alpha=float(model.params['const']),
                    beta=float(model.params[factor]), t_stat=float(model.tvalues[factor]),
                    p_value=float(model.pvalues[factor]), observations=int(model.nobs),
                    r_squared=float(model.rsquared)))

        def select_block(columns, stage):
            model, selected, log = backward_select(y, data[columns])
            selection.extend(dict(stagger=stagger, ticker=ticker, stage=stage, **row)
                             for row in log)
            apply(model, selected, stage)
            return selected

        checkpoint('Original')
        if home is not None:
            apply(fit(y, data[[home]]), [home], 'Home FX')
        checkpoint('Home FX')
        selected_fx = select_block([c for c in currencies if c != home], 'Other FX')
        checkpoint('Other FX')
        apply(fit(y, data[['SPY']]), ['SPY'], 'SPY')
        checkpoint('SPY')
        apply(fit(y, data[[primary]]), [primary], 'Primary sector')
        checkpoint('Primary sector')
        selected_sectors = select_block([s for s in SECTORS.values() if s != primary], 'Other sectors')
        checkpoint('Other sectors')
        error = float((original - contribution - y).abs().max())
        if error > 1e-10:
            raise AssertionError(f'Return reconstruction failed: {ticker}')
        stds = np.array([r['residual_std'] for r in progress[-len(STAGES):]])
        if np.any(np.diff(stds) > 1e-12):
            raise AssertionError(f'Residual standard deviation increased between blocks: {ticker}')
        exposure['alpha_total'] = alpha_total
        exposures.append(exposure)
        residuals[ticker] = y
        diagnostics.append(dict(stagger=stagger, ticker=ticker, company=company.company_name,
            currency=company.currency, primary_sector=primary, observations=len(data),
            start=data.index.min(), end=data.index.max(), original_std=float(original.std(ddof=1)),
            residual_std=float(y.std(ddof=1)), std_reduction=1-y.std(ddof=1)/original.std(ddof=1),
            variance_reduction=1-y.var(ddof=1)/original.var(ddof=1),
            reconstruction_r_squared=1-float((y**2).sum())/float(((original-original.mean())**2).sum()),
            retained_fx=','.join(selected_fx), additional_sectors=','.join(selected_sectors),
            reconstruction_error=error))

    # One common date sample gives a PSD covariance matrix, unlike pairwise deletion.
    common = residuals.dropna(how='any')
    if len(common) < 2:
        raise ValueError('At least two shared dates are required for covariance/PCA')
    residual_cov = common.cov(ddof=1)
    factor_cov = factors.loc[common.index].cov(ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(residual_cov.to_numpy())
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if eigenvalues.min() < -1e-10:
        raise AssertionError('Residual covariance is not positive semidefinite')
    eigenvalues = np.maximum(eigenvalues, 0)
    for j in range(len(eigenvalues)):
        if eigenvectors[np.abs(eigenvectors[:, j]).argmax(), j] < 0:
            eigenvectors[:, j] *= -1
    pcs = [f'PC{i+1}' for i in range(len(eigenvalues))]
    scores = (common-common.mean()).to_numpy() @ eigenvectors
    np.testing.assert_allclose(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T,
                               residual_cov, atol=1e-10)
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(len(pcs)), atol=1e-10)
    spectrum = pd.DataFrame(dict(component=pcs, eigenvalue=eigenvalues,
        explained_share=eigenvalues/eigenvalues.sum(),
        cumulative_share=np.cumsum(eigenvalues)/eigenvalues.sum()))
    tables = dict(stage_coefficients=pd.DataFrame(records), factor_exposures=pd.DataFrame(exposures),
        diagnostics=pd.DataFrame(diagnostics), selection_log=pd.DataFrame(selection),
        residual_progress=pd.DataFrame(progress), residual_covariance=residual_cov,
        factor_covariance=factor_cov, pca_eigenvalues=spectrum,
        pca_eigenvectors=pd.DataFrame(eigenvectors, index=common.columns, columns=pcs),
        pca_scores=pd.DataFrame(scores, index=common.index, columns=pcs))
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        indexed = name in {'residual_covariance', 'factor_covariance', 'pca_eigenvectors', 'pca_scores'}
        frame.to_csv(output/f'{name}.csv', index=indexed,
                     index_label='date' if name == 'pca_scores' else 'factor_or_stock')
    residuals.to_csv(output/f'{stagger}_residuals.csv', index_label='date')
    for stage, frame in stage_returns.items():
        frame.to_csv(output/f"stage_{stage.lower().replace(' ', '_')}.csv", index_label='date')
    factors.to_csv(output/'factor_returns.csv', index_label='date')
    returns[companies.ticker].to_csv(output/'stock_returns.csv', index_label='date')
    companies.to_csv(output/'companies.csv', index=False)
    metadata = dict(method='Home FX, backward FX selection, SPY, primary sector, backward sector selection',
        stagger=stagger, stocks=len(companies), factors=factor_names, threshold=2.0,
        common_observations=len(common), common_start=str(common.index.min().date()),
        common_end=str(common.index.max().date()), source='data/companies_and_staggered_returns.xlsx',
        common_dates=[str(d.date()) for d in common.index])
    (output/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Completed {len(companies)} stocks, {len(factor_names)} factors, {len(common)} common dates')
    return tables, {stagger: residuals}


if __name__ == '__main__':
    run()
