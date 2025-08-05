import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import re

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample indices - replace with your actual indices
AVAILABLE_INDICES = [
    'USD_SOFR', 'EUR_ESTR', 'GBP_SONIA', 'JPY_TONAR',
    'CAD_CORRA', 'AUD_AONIA', 'CHF_SARON', 'SEK_SWESTR'
]

# Placeholder function - replace with your actual function
def your_repo_function(indices_list, dates_list, start_date):
    """
    Placeholder for your actual repo curve function.
    Replace this with your real function that returns a DataFrame.
    
    Parameters:
    - indices_list: list of selected indices
    - dates_list: list of maturity dates
    - start_date: start date for the analysis
    
    Returns:
    - DataFrame with curves for each index and maturity
    """
    # This is a mock implementation - replace with your actual code
    import numpy as np
    
    data = []
    for idx in indices_list:
        for date in dates_list:
            # Generate sample curve data
            x_values = np.linspace(0, 365, 100)  # Days
            y_values = np.random.normal(0.02, 0.005, 100) + np.random.random() * 0.01
            
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                data.append({
                    'Index': idx,
                    'Maturity': str(date),
                    'Days': x,
                    'Rate': y,
                    'Curve_ID': f"{idx}_{date}"
                })
    
    return pd.DataFrame(data)

def parse_maturity_input(input_text):
    """
    Parse maturity input and convert to list of dates or months.
    Supports:
    - Dates in YYYY-MM-DD format
    - Integers (months)
    - Comma-separated values
    """
    if not input_text.strip():
        return []
    
    items = [item.strip() for item in input_text.split(',')]
    parsed_items = []
    
    for item in items:
        if not item:
            continue
            
        # Try to parse as date (YYYY-MM-DD)
        try:
            parsed_date = datetime.strptime(item, '%Y-%m-%d').date()
            parsed_items.append(parsed_date)
        except ValueError:
            # Try to parse as integer (months)
            try:
                months = int(item)
                parsed_items.append(months)
            except ValueError:
                # Skip invalid entries
                continue
    
    return parsed_items

# Define the app layout
app.layout = html.Div([
    html.H1("Repo Curve Visualization", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        # Index selection
        html.Div([
            html.Label("Select Indices:", 
                      style={'fontWeight': 'bold', 'marginBottom': 10}),
            dcc.Dropdown(
                id='index-dropdown',
                options=[{'label': idx, 'value': idx} for idx in AVAILABLE_INDICES],
                value=[AVAILABLE_INDICES[0]],  # Default selection
                multi=True,
                placeholder="Select one or more indices..."
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Start date input
        html.Div([
            html.Label("Start Date:", 
                      style={'fontWeight': 'bold', 'marginBottom': 10}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date=datetime.now().date(),
                display_format='YYYY-MM-DD'
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'marginBottom': 20}),
    
    # Maturity input
    html.Div([
        html.Label("Maturities (comma-separated):", 
                  style={'fontWeight': 'bold', 'marginBottom': 10}),
        html.Br(),
        html.Small("Enter dates (YYYY-MM-DD) or months (integers), separated by commas. Example: 2024-12-31, 6, 2025-06-15, 12"),
        dcc.Textarea(
            id='maturity-input',
            value='3, 6, 12, 2024-12-31',  # Default values
            placeholder='Enter maturities: 3, 6, 12, 2024-12-31, 24...',
            style={'width': '100%', 'height': 60, 'marginTop': 5}
        )
    ], style={'marginBottom': 20}),
    
    # Run button
    html.Div([
        html.Button('Generate Curves', 
                   id='run-button', 
                   n_clicks=0,
                   style={
                       'backgroundColor': '#007bff',
                       'color': 'white',
                       'padding': '10px 20px',
                       'border': 'none',
                       'borderRadius': '5px',
                       'cursor': 'pointer',
                       'fontSize': '16px'
                   })
    ], style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Status message
    html.Div(id='status-message', style={'marginBottom': 20}),
    
    # Plot
    dcc.Graph(id='repo-curves-plot', style={'height': '600px'})
    
], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})

# Callback for updating the plot
@app.callback(
    [Output('repo-curves-plot', 'figure'),
     Output('status-message', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('index-dropdown', 'value'),
     State('maturity-input', 'value'),
     State('start-date-picker', 'date')]
)
def update_plot(n_clicks, selected_indices, maturity_input, start_date):
    if n_clicks == 0:
        # Initial empty plot
        fig = go.Figure()
        fig.update_layout(
            title="Select parameters and click 'Generate Curves' to view the repo curves",
            xaxis_title="Days",
            yaxis_title="Rate",
            template="plotly_white"
        )
        return fig, ""
    
    # Validate inputs
    if not selected_indices:
        error_msg = html.Div("Please select at least one index.", 
                           style={'color': 'red', 'textAlign': 'center'})
        return go.Figure(), error_msg
    
    # Parse maturity input
    parsed_maturities = parse_maturity_input(maturity_input)
    if not parsed_maturities:
        error_msg = html.Div("Please enter valid maturities (dates in YYYY-MM-DD format or integers for months).", 
                           style={'color': 'red', 'textAlign': 'center'})
        return go.Figure(), error_msg
    
    try:
        # Convert start_date string to date object if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        # Call your function
        df = your_repo_function(selected_indices, parsed_maturities, start_date)
        
        # Create the plot
        fig = go.Figure()
        
        # Generate colors for different curves
        colors = px.colors.qualitative.Set1
        color_idx = 0
        
        # Plot each curve
        for curve_id in df['Curve_ID'].unique():
            curve_data = df[df['Curve_ID'] == curve_id]
            
            fig.add_trace(go.Scatter(
                x=curve_data['Days'],
                y=curve_data['Rate'],
                mode='lines',
                name=curve_id,
                line=dict(color=colors[color_idx % len(colors)], width=2)
            ))
            color_idx += 1
        
        # Update layout
        fig.update_layout(
            title=f"Repo Curves - Start Date: {start_date}",
            xaxis_title="Days",
            yaxis_title="Rate",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        success_msg = html.Div(
            f"Successfully generated curves for {len(selected_indices)} indices and {len(parsed_maturities)} maturities.",
            style={'color': 'green', 'textAlign': 'center'}
        )
        
        return fig, success_msg
        
    except Exception as e:
        error_msg = html.Div(
            f"Error generating curves: {str(e)}",
            style={'color': 'red', 'textAlign': 'center'}
        )
        return go.Figure(), error_msg

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)






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
