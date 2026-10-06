"""Run the six sequential stock-factor regression steps for the assigned stagger."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

SECTORS = dict(zip(
    ['Communication Services', 'Consumer Discretionary', 'Consumer Staples',
     'Energy', 'Financials', 'Health Care', 'Industrials', 'Materials',
     'Real Estate', 'Technology', 'Utilities'],
    ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']))
ROOT = Path(__file__).resolve().parents[1]


def section(raw, title):
    """Locate a titled Excel table without depending on fixed row offsets."""
    rows = raw.index[raw.iloc[:, 0].eq(title)]
    if len(rows) != 1:
        raise ValueError(f'Expected one section: {title}')
    start = int(rows[0]) + 1
    end = start + 1
    while end < len(raw) and pd.notna(raw.iloc[end, 0]):
        end += 1
    frame = raw.iloc[start + 1:end].copy()
    frame.columns = raw.iloc[start].tolist()
    frame = frame.loc[:, frame.columns.notna()]
    frame['date'] = pd.to_datetime(frame['date'])
    return frame.set_index('date').apply(pd.to_numeric, errors='raise')


def fit(y, x):
    return sm.OLS(y, sm.add_constant(x, has_constant='add')).fit()


def run(source, output, threshold=2.0, stagger="stagger_1"):
    companies = pd.read_excel(source, sheet_name='Companies')
    companies = companies.dropna(subset=['sector', 'currency']).drop_duplicates('ticker')
    records, diagnostics, selection, residual_tables, exposures = [], [], [], {}, []
    for stagger in [stagger]:
        raw = pd.read_excel(source, sheet_name=stagger, header=None)
        returns = section(raw, 'Time-weighted four-week log returns')
        sectors = section(raw, 'Sector regression residual returns')
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
            exposure = {'stagger': stagger, 'ticker': ticker,
                        **{f'beta_{f}': 0.0 for f in ['MXN/USD','SGD/USD','CAD/USD','DKK/USD','EUR/USD','AUD/USD','SPY', *SECTORS.values()]}}

            def stage(step, factor, conditional=False):
                nonlocal y, contribution
                model = fit(y, data[[factor]])
                keep = not conditional or abs(model.tvalues[factor]) > threshold
                records.append(dict(stagger=stagger, ticker=ticker, step=step,
                    factor=factor, alpha=model.params['const'], beta=model.params[factor],
                    t_stat=model.tvalues[factor], p_value=model.pvalues[factor],
                    retained=keep, applied_beta=model.params[factor] if keep else 0.0,
                    observations=int(model.nobs), r_squared=model.rsquared))
                if keep:
                    contribution += model.fittedvalues
                    y = model.resid.copy()
                    exposure[f'beta_{factor.removesuffix("_residual")}'] = model.params[factor]

            if foreign:
                stage(1, currency)
            stage(2 if foreign else 3, 'AUD/USD', conditional=True)
            stage(4, 'SPY')
            stage(5, primary + '_residual')
            # Forward selection on the stage-5 response, jointly refitting the
            # selected additional sectors at each iteration. All retained
            # additional-sector coefficients must pass the absolute-t cutoff.
            response = y.copy()
            selected = []
            remaining = sorted(f'{s}_residual' for s in SECTORS.values() if s != primary)
            while remaining:
                candidates = []
                for candidate in remaining:
                    model = fit(response, data[selected + [candidate]])
                    score = abs(model.tvalues[candidate])
                    admissible = all(abs(model.tvalues[f]) > threshold for f in selected + [candidate])
                    selection.append(dict(stagger=stagger, ticker=ticker,
                        iteration=len(selected) + 1, candidate=candidate,
                        t_stat=model.tvalues[candidate], admissible=admissible))
                    if admissible:
                        candidates.append((score, candidate))
                if not candidates:
                    break
                _, chosen = max(candidates, key=lambda pair: (pair[0], pair[1]))
                selected.append(chosen)
                remaining.remove(chosen)
            if selected:
                model = fit(response, data[selected])
                contribution += model.fittedvalues
                y = model.resid.copy()
                for factor in selected:
                    records.append(dict(stagger=stagger, ticker=ticker, step=6,
                        factor=factor, alpha=model.params['const'], beta=model.params[factor],
                        t_stat=model.tvalues[factor], p_value=model.pvalues[factor],
                        retained=True, applied_beta=model.params[factor],
                        observations=int(model.nobs), r_squared=model.rsquared))
                    exposure[f'beta_{factor.removesuffix("_residual")}'] = model.params[factor]
            error = float((original - contribution - y).abs().max())
            if error > 1e-10:
                raise AssertionError('Return reconstruction failed')
            exposure['alpha_total'] = float(contribution.mean() - sum(
                exposure[f'beta_{f.removesuffix("_residual")}'] * data[f].mean()
                for f in ([currency] if foreign else []) + ['AUD/USD','SPY',primary+'_residual'] + selected))
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
              'diagnostics': pd.DataFrame(diagnostics), 'selection_log': pd.DataFrame(selection)}
    with pd.ExcelWriter(output / 'stock_factor_results.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            table.to_csv(output / f'{name}.csv', index=False)
        for name, table in residual_tables.items():
            table.to_excel(writer, sheet_name=name + '_residuals', index_label='date')
            table.to_csv(output / f'{name}_residuals.csv', index_label='date')
    print(f'Completed {len(diagnostics)} stock/stagger models; results: {output}')
    return tables, residual_tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/companies_and_staggered_returns.xlsx')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/results')
    parser.add_argument('--threshold', type=float, default=2.0)
    parser.add_argument('--stagger', choices=[f'stagger_{i}' for i in range(1, 5)], default='stagger_1')
    args = parser.parse_args()
    run(args.source, args.output, args.threshold, args.stagger)
