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
