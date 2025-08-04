
import pandas as pd
import numpy as np
from typing import List, Union
from datetime import datetime
from xbbg import blp  # suppose xbbg est installé et configuré

def get_fx_spot_rates(currencies: List[str],
                      asof_date: Union[str, datetime],
                      multipliers_df: pd.DataFrame,
                      ref_currency: str = "USD") -> List[float]:
    """
    Récupère pour chaque currency de la liste son spot contre ref_currency à la date donnée,
    applique le multiplicateur via merge, et retourne la liste des rates ajustés (dans l'ordre),
    sans les noms.

    Params:
        currencies: ex ["EUR", "KWD", ...]
        asof_date: "YYYY-MM-DD" ou datetime
        multipliers_df: DataFrame contenant soit :
            - format long : colonnes ["currency", "multiplier"]
            - format wide : colonnes par devise (ex "KWD", "EUR") avec une ligne de multiplicateurs
        ref_currency: devise de référence, default "USD"
    Returns:
        List[float]: spot rates ajustés (avec multiplicateur) dans l’ordre de `currencies`.
                     Si une devise manque, np.nan est placé.
    """
    # Normaliser date
    if isinstance(asof_date, datetime):
        date_str = asof_date.strftime("%Y-%m-%d")
    else:
        # on suppose une string ISO-like
        # parfois xbbg accepte "YYYY-MM-DD"
        date_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")

    # Préparer dataframe de devises
    df = pd.DataFrame({"currency": currencies})

    # Préparer les multiplicateurs : transformer wide -> long si nécessaire
    if set(multipliers_df.columns) >= {"currency", "multiplier"}:
        mult_long = multipliers_df[["currency", "multiplier"]].copy()
    else:
        # wide format : une ligne attendue, colonnes = codes devises
        # on melt pour obtenir long
        try:
            mult_long = multipliers_df.reset_index(drop=True).melt(var_name="currency", value_name="multiplier")
        except Exception:
            raise ValueError("multipliers_df format non reconnu. Il doit être either long (currency/multiplier) or wide (one row, columns=currencies).")

    # Merge multiplicateur dans l'ordre des currencies
    df = df.merge(mult_long, on="currency", how="left")
    df["multiplier"] = df["multiplier"].fillna(1.0).astype(float)

    # Construire tickers (ex: "EURUSD Curncy")
    df["ticker"] = df["currency"] + ref_currency + " Curncy"

    # Appel BDH : tentative groupée
    tickers = df["ticker"].tolist()
    try:
        raw = blp.bdh(tickers, flds=["PX_LAST"], start_date=date_str, end_date=date_str)
    except Exception:
        raw = pd.DataFrame()

    spots = []
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        spot = None
        # Essayer de prendre dans bdh
        try:
            if not raw.empty:
                # xbbg bdh retourne colonnes simples si 1 field: (ticker,) or (ticker, field) depending version
                if (ticker, "PX_LAST") in raw.columns:
                    series = raw[(ticker, "PX_LAST")]
                elif ticker in raw.columns:
                    series = raw[ticker]
                else:
                    series = pd.Series(dtype=float)
                if not series.empty:
                    val = series.iloc[0]
                    if pd.notna(val):
                        spot = float(val)
        except Exception:
            spot = None

        # Fallback bdp si besoin
        if spot is None or (isinstance(spot, float) and np.isnan(spot)):
            try:
                bdp_res = blp.bdp(ticker, flds=["PX_LAST"])
                if "PX_LAST" in bdp_res.columns:
                    val = bdp_res.loc[0, "PX_LAST"]
                    if pd.notna(val):
                        spot = float(val)
                elif ticker in bdp_res.columns:
                    val = bdp_res.loc[0, ticker]
                    if pd.notna(val):
                        spot = float(val)
            except Exception:
                spot = None

        if spot is None:
            spot = np.nan

        # Appliquer multiplicateur
        adjusted = spot * row["multiplier"] if not pd.isna(spot) else np.nan
        spots.append(adjusted)

    return spots







import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

# --------- Utilities ---------
def is_quarter_end(dt_index):
    quarters = pd.date_range(start=dt_index.min(), end=dt_index.max(), freq='Q')
    return dt_index.normalize().isin(quarters.normalize()).astype(int)

def rolling_vol(series, window=5):
    return series.rolling(window).std()

# --------- Analysis pipeline ---------
def compute_basic_stats(daily):
    df = daily.copy()
    df['quarter_end'] = is_quarter_end(df.index)
    # Global correlation
    corr_global = df['EURUSD_ret'].corr(df['SPX_ret'])
    # Correlation conditional
    corr_qe = df.loc[df['quarter_end'] == 1, 'EURUSD_ret'].corr(df.loc[df['quarter_end'] == 1, 'SPX_ret'])
    corr_non_qe = df.loc[df['quarter_end'] == 0, 'EURUSD_ret'].corr(df.loc[df['quarter_end'] == 0, 'SPX_ret'])
    return df, corr_global, corr_qe, corr_non_qe

def plot_correlation_scatter(df):
    plt.figure()
    plt.scatter(df['SPX_ret'], df['EURUSD_ret'], alpha=0.6)
    plt.xlabel("SPX return (16:00→20:59)")
    plt.ylabel("EURUSD return (16:00→20:59)")
    plt.title("Scatter: SPX vs EURUSD returns")
    plt.grid(True)
    plt.tight_layout()

def plot_distributions(df):
    # joint conditional on sign of SPX
    plt.figure()
    df_pos = df[df['SPX_ret'] > 0]
    df_neg = df[df['SPX_ret'] < 0]
    plt.hist(df_pos['EURUSD_ret'], bins=50, alpha=0.5, label='SPX > 0')
    plt.hist(df_neg['EURUSD_ret'], bins=50, alpha=0.5, label='SPX < 0')
    plt.legend()
    plt.xlabel("EURUSD return")
    plt.title("EURUSD return distribution conditional on SPX sign")
    plt.grid(True)
    plt.tight_layout()

def regression_linear(df):
    # EURUSD_ret ~ SPX_ret + quarter_end
    X = pd.DataFrame({
        'SPX_ret': df['SPX_ret'],
        'quarter_end': is_quarter_end(df.index)
    })
    X = sm.add_constant(X)
    y = df['EURUSD_ret']
    model = sm.OLS(y, X).fit()
    print("=== Linear regression EURUSD_ret ~ SPX_ret + quarter_end ===")
    print(model.summary())
    return model

def logistic_directional(df):
    # target: EURUSD up or down
    df = df.copy()
    df['fx_up'] = (df['EURUSD_ret'] > 0).astype(int)
    df['quarter_end'] = is_quarter_end(df.index)
    # features: SPX_ret, quarter_end, recent vol of EURUSD
    df['vol_fx_5d'] = rolling_vol(df['EURUSD_ret'], window=5)
    feat = df[['SPX_ret', 'quarter_end', 'vol_fx_5d']].dropna()
    label = df.loc[feat.index, 'fx_up']
    X = feat.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    all_true, all_pred_prob = [], []
    for train_idx, test_idx in tscv.split(Xs):
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train, y_test = label.iloc[train_idx], label.iloc[test_idx]
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        all_true.append(y_test.values)
        all_pred_prob.append(proba)
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_pred_prob)
    y_pred_label = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    print(f"=== Logistic direction model (EURUSD up) ===\nROC AUC: {auc:.3f}")
    print(classification_report(y_true, y_pred_label))
    return clf, scaler, auc

def event_study_quarter_ends(df, window=5):
    df = df.copy()
    df['quarter_end'] = is_quarter_end(df.index)
    fx_col = 'EURUSD_ret'
    eq_col = 'SPX_ret'
    events = df[df['quarter_end'] == 1].index
    paths_fx = []
    paths_eq = []
    for ev in events:
        start = ev - pd.Timedelta(days=window)
        end = ev + pd.Timedelta(days=window)
        seg = df.loc[start:end, [fx_col, eq_col]].copy()
        if len(seg) < (2 * window + 1):
            continue
        seg = seg.reset_index()
        seg['day_offset'] = (seg['index'] - ev).days
        seg = seg.set_index('day_offset')
        paths_fx.append(seg[fx_col])
        paths_eq.append(seg[eq_col])
    if not paths_fx:
        print("No full quarter-end events in window.")
        return None
    mean_fx = pd.concat(paths_fx, axis=1).mean(axis=1)
    mean_eq = pd.concat(paths_eq, axis=1).mean(axis=1)
    summary = pd.DataFrame({fx_col: mean_fx, eq_col: mean_eq})
    # plot
    plt.figure()
    summary.plot(marker='o')
    plt.title(f"Event study around quarter ends (±{window} days)")
    plt.xlabel("Day offset from quarter-end")
    plt.ylabel("Average return")
    plt.grid(True)
    plt.tight_layout()
    return summary

def simulate_hedge(df):
    """
    Hedge naïf : on fixe le rate EURUSD à 16h (ici on assume return measure implies price change)
    Simuler PnL d'un hedge spot simple et comparer à exposition non-hedgée.
    On suppose une exposition de 1 unit USD à vendre à 21h : si EURUSD baisse, perte si non couvert.
    Hedge naive = short EURUSD à 16h, realized à 21h.
    """
    df = df.copy()
    # PnL non-hedged en EUR d'une vente de 1 USD à 21h : si EURUSD_ret = (P21/P16)-1, 
    # alors taux passe de S16 à S21, et vendre USD à 21h convertit via S21.
    # Hedge spot à 16h fixe le taux S16, donc perte = différence entre S21 et S16.
    # On mesure en PnL relatif: hedge error = (1 / (1 + EURUSD_ret)) - 1 approximé par -EURUSD_ret
    df['hedge_error'] = -df['EURUSD_ret']  # simplification
    df['abs_error'] = df['hedge_error'].abs()
    # Performance summary
    print("=== Hedge naïf summary ===")
    print("Mean error:", df['hedge_error'].mean())
    print("Std error:", df['hedge_error'].std())
    # Conditional on quarter-end
    df['quarter_end'] = is_quarter_end(df.index)
    print("Mean error on quarter-end days:", df.loc[df['quarter_end'] == 1, 'hedge_error'].mean())
    print("Mean error off quarter-end:", df.loc[df['quarter_end'] == 0, 'hedge_error'].mean())

    # Plot distribution
    plt.figure()
    plt.hist(df['hedge_error'], bins=50)
    plt.title("Distribution du hedge error (naïf) EURUSD")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    return df

def summary_plots(df, corr_global, corr_qe, corr_non_qe):
    # correlation bar
    plt.figure()
    labels = ['Global', 'Quarter-end', 'Non-quarter-end']
    values = [corr_global, corr_qe, corr_non_qe]
    plt.bar(labels, values)
    plt.ylabel("Corr(EURUSD, SPX)")
    plt.title("Correlation comparison")
    plt.grid(True, axis='y')
    plt.tight_layout()

# --------- Entrée : ton DataFrame daily ---------
# Supposons que tu as déjà un DataFrame `daily` avec columns ['EURUSD_ret', 'SPX_ret']
# Exemple d'appel :
# daily = pd.read_csv("tes_returns.csv", parse_dates=True, index_col=0)
# daily, corr_global, corr_qe, corr_non_qe = compute_basic_stats(daily)
# plot_correlation_scatter(daily)
# plot_distributions(daily)
# summary_plots(daily, corr_global, corr_qe, corr_non_qe)
# lin_model = regression_linear(daily)
# log_model, scaler, auc = logistic_directional(daily)
# es_summary = event_study_quarter_ends(daily, window=3)
# hedge_df = simulate_hedge(daily)
# plt.show()

# --------- Optionnel : sauvegarde des résultats ---------
# es_summary.to_csv("event_study_qe.csv")
# hedge_df.to_csv("hedge_simulation.csv")







import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- INPUT: y = spread, X = tariff time series ---
# y and X are both pandas Series with daily datetime index
# Example placeholders:
# y = pd.Series(..., name="spread")
# X = pd.Series(..., name="tariffs")

# --- Step 1: Define tariff-relevant periods (e.g. Taiwan tariffs) ---
relevant_periods = (
    ((X.index >= "2019-01-01") & (X.index <= "2022-12-31")) |
    ((X.index >= "2025-01-01") & (X.index <= "2025-12-31"))
)

# --- Step 2: Filter data to tariff-relevant periods only ---
X_relevant = X[relevant_periods]
y_relevant = y[relevant_periods]

# --- Step 3: Align and drop missing values ---
df = pd.concat([y_relevant, X_relevant], axis=1).dropna()
df.columns = ['spread', 'tariffs']

# --- Step 4: Run regression ---
X_reg = sm.add_constant(df['tariffs'])  # add intercept
model = sm.OLS(df['spread'], X_reg)
results = model.fit()

# --- Step 5: Output summary ---
print(results.summary())

# --- Step 6: Optional plot ---
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['spread'], label='Actual Spread', color='black', linewidth=1)
plt.plot(df.index, results.fittedvalues, label='Fitted Spread', color='blue', linestyle='--')
plt.title('Fitted vs Actual Spread during Tariff-Relevant Periods')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.tight_layout()
plt.show()







import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Simulated input: replace this with your actual function ---
def get_div_yield(start_date: str, tenor: str) -> pd.DataFrame:
    """
    Simulates a dividend yield DataFrame.
    Replace this function with your real data loader.
    
    Returns:
        index = assets
        columns = monthly dates
        values = annualized dividend yield
    """
    dates = pd.date_range(start=start_date, periods=6, freq='M')
    assets = ['Asset_A', 'Asset_B', 'Asset_C']
    np.random.seed(hash(tenor) % 2**32)
    data = np.random.uniform(0.015, 0.035, size=(len(assets), len(dates)))
    return pd.DataFrame(data, index=assets, columns=dates)

# --- Parameters ---
tenors = ['3M', '6M', '9M', '12M']
asset_to_plot = 'Asset_A'
start_date = "2025-02-28"

# --- Step 1: Gather and reshape all data ---
all_data = []

for tenor in tenors:
    df = get_div_yield(start_date=start_date, tenor=tenor)
    df['Asset'] = df.index
    df_long = df.melt(id_vars='Asset', var_name='Date', value_name='Annualized_Yield')
    df_long['Tenor'] = tenor
    all_data.append(df_long)

# Combine all tenors into one tidy DataFrame
df_full = pd.concat(all_data, ignore_index=True)

# --- Step 2: Preprocess for 3D plotting ---
df_full['Date'] = pd.to_datetime(df_full['Date'])
df_full['Tenor_Num'] = df_full['Tenor'].str.replace('M', '').astype(int)

# Filter to one asset
df_asset = df_full[df_full['Asset'] == asset_to_plot].copy()

# Create grid: rows = tenors, columns = dates
pivoted = df_asset.pivot(index='Tenor_Num', columns='Date', values='Annualized_Yield')
pivoted = pivoted.sort_index().sort_index(axis=1)

# --- Step 3: 3D Plot ---
fig = go.Figure(data=[go.Surface(
    z=pivoted.values,
    x=[d.strftime('%Y-%m-%d') for d in pivoted.columns],  # X = Date
    y=pivoted.index,  # Y = Tenor in months
    colorscale='Viridis',
    colorbar_title='Yield'
)])

fig.update_layout(
    title=f'3D Surface of Annualized Dividend Yield – {asset_to_plot}',
    scene=dict(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Tenor (months)'),
        zaxis=dict(title='Annualized Yield')
    ),
    autosize=True,
    height=700
)

fig.show()
