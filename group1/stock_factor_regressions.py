"""Run the six sequential stock-factor regression steps for the assigned stagger."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

SECTORS = {
    'Communication Services': 'XLC',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Financials': 'XLF',
    'Health Care': 'XLV',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Technology': 'XLK',
    'Utilities': 'XLU',
}
ROOT = Path(__file__).resolve().parents[1]


def fit(y, x):
    return sm.OLS(y, sm.add_constant(x, has_constant='add')).fit()


def run():
    source = ROOT / "data/companies_and_staggered_returns.xlsx"
    output = ROOT / "group1/results"
    threshold = 2.0
    stagger = "stagger_1"
    companies = pd.read_excel(source, sheet_name='Companies')
    companies = companies.dropna(subset=['sector', 'currency']).drop_duplicates('ticker')
    records, diagnostics, selection, residual_tables, exposures = [], [], [], {}, []
    residual_progress = []
    # Excel row 255 starts weighted returns; row 339 starts sector residuals.
    returns = pd.read_excel(source, sheet_name=stagger, skiprows=254, nrows=80)
    sectors = pd.read_excel(source, sheet_name=stagger, skiprows=338, nrows=80)
    # The sector section is narrower than the price section in this sheet.
    sectors = sectors.loc[:, ~sectors.columns.astype(str).str.startswith('Unnamed:')]
    returns['date'] = pd.to_datetime(returns['date'])
    sectors['date'] = pd.to_datetime(sectors['date'])
    returns = returns.set_index('date')
    sectors = sectors.set_index('date')
    residual_table = pd.DataFrame(index=returns.index)
    for company in companies.itertuples(index=False):
        ticker, primary = company.ticker, SECTORS[company.sector]
        currency = f'{company.currency}/USD'
        foreign = company.currency != 'USD'
        required = [ticker, 'AUD/USD', 'SPY'] + ([currency] if foreign else [])
        data = returns[required].join(sectors).replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < len(SECTORS) + 5:
            raise ValueError(f'Insufficient observations: {stagger}, {ticker}')
        y = data[ticker].copy()
        original = y.copy()
        contribution = pd.Series(0.0, index=y.index)
        def record_residual_stage(stage, retained):
            residual_progress.append(dict(
                stagger=stagger, ticker=ticker, stage=stage, retained=retained,
                observations=len(y), residual_std=float(y.std(ddof=1))))

        record_residual_stage('Original', False)
        if not foreign:
            record_residual_stage('Home FX', False)
        exposure = {'stagger': stagger, 'ticker': ticker,
                    **{f'beta_{f}': 0.0 for f in ['MXN/USD','SGD/USD','CAD/USD','DKK/USD','EUR/USD','AUD/USD','SPY', *SECTORS.values()]}}
    
        # Steps 1-5: one regression at a time, passing residuals forward.
        steps = []
        if foreign:
            steps.append((1, currency, False))
        aud_step = 2 if foreign else 3
        steps.append((aud_step, 'AUD/USD', True))
        steps.append((4, 'SPY', False))
        steps.append((5, primary + '_residual', False))
        alpha_total = 0.0

        for step, factor, conditional in steps:
            model = fit(y, data[[factor]])
            beta = model.params[factor]
            t_stat = model.tvalues[factor]
            keep = not conditional or abs(t_stat) > threshold
            records.append({
                'stagger': stagger, 'ticker': ticker, 'step': step,
                'factor': factor, 'alpha': model.params['const'], 'beta': beta,
                't_stat': t_stat, 'p_value': model.pvalues[factor],
                'retained': keep, 'applied_beta': beta if keep else 0.0,
                'observations': int(model.nobs), 'r_squared': model.rsquared,
            })
            if keep:
                contribution = contribution + model.fittedvalues
                y = model.resid.copy()
                alpha_total = alpha_total + model.params['const']
                factor_name = factor.replace('_residual', '')
                exposure['beta_' + factor_name] = beta
            stage_name = {1: 'Home FX', 2: 'AUD', 3: 'AUD', 4: 'SPY', 5: 'Primary sector'}[step]
            record_residual_stage(stage_name, keep)

        # Forward selection on the stage-5 response, jointly refitting the
        # selected additional sectors at each iteration. All retained
        # additional-sector coefficients must pass the absolute-t cutoff.
        response = y.copy()
        selected = []
        remaining = sorted(f'{s}_residual' for s in SECTORS.values() if s != primary)
        while remaining:
            best_score = threshold
            chosen = None
            for candidate in remaining:
                model = fit(response, data[selected + [candidate]])
                score = abs(model.tvalues[candidate])
                admissible = all(abs(model.tvalues[f]) > threshold for f in selected + [candidate])
                selection.append(dict(stagger=stagger, ticker=ticker,
                    iteration=len(selected) + 1, candidate=candidate,
                    t_stat=model.tvalues[candidate], admissible=admissible))
                if admissible and score >= best_score:
                    best_score = score
                    chosen = candidate
            if chosen is None:
                break
            selected.append(chosen)
            remaining.remove(chosen)
        if selected:
            model = fit(response, data[selected])
            contribution += model.fittedvalues
            alpha_total += model.params["const"]
            y = model.resid.copy()
            for factor in selected:
                records.append(dict(stagger=stagger, ticker=ticker, step=6,
                    factor=factor, alpha=model.params['const'], beta=model.params[factor],
                    t_stat=model.tvalues[factor], p_value=model.pvalues[factor],
                    retained=True, applied_beta=model.params[factor],
                    observations=int(model.nobs), r_squared=model.rsquared))
                exposure[f'beta_{factor.removesuffix("_residual")}'] = model.params[factor]
        record_residual_stage('Additional sectors', bool(selected))
        error = float((original - contribution - y).abs().max())
        if error > 1e-10:
            raise AssertionError('Return reconstruction failed')
        exposure['alpha_total'] = alpha_total
        exposures.append(exposure)
        residual_table[ticker] = y
        diagnostics.append(dict(stagger=stagger, ticker=ticker, currency=company.currency,
            primary_sector=primary, observations=len(data), start=data.index.min(),
            end=data.index.max(), residual_std=y.std(ddof=1),
            additional_sectors=','.join(f.removesuffix('_residual') for f in selected),
            reconstruction_error=error))
    residual_tables[stagger] = residual_table
    output.mkdir(parents=True, exist_ok=True)
    tables = {'stage_coefficients': pd.DataFrame(records), 'factor_exposures': pd.DataFrame(exposures),
              'diagnostics': pd.DataFrame(diagnostics), 'selection_log': pd.DataFrame(selection),
              'residual_progress': pd.DataFrame(residual_progress)}
    with pd.ExcelWriter(output / 'stock_factor_results.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            table.to_csv(output / f'{name}.csv', index=False)
        for name, table in residual_tables.items():
            table.to_excel(writer, sheet_name=name + '_residuals', index_label='date')
            table.to_csv(output / f'{name}_residuals.csv', index_label='date')
    print(f'Completed {len(diagnostics)} stock/stagger models; results: {output}')
    return tables, residual_tables


if __name__ == "__main__":
    run()
