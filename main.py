import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime
from datetime import date, timedelta
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample data - replace with your actual indices and maturities
AVAILABLE_INDICES = [
    'SPX', 'NDX', 'RUT', 'VIX', 'DJI', 'ES', 'NQ', 'YM', 'RTY'
]

# Generate quarterly expiry dates for the next 2 years
def generate_quarterly_expiries():
    expiries = []
    current_year = datetime.datetime.now().year
    
    for year in range(current_year, current_year + 3):
        for month in [3, 6, 9, 12]:  # March, June, September, December
            # Third Friday of the month (typical futures expiry)
            first_day = datetime.date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            expiries.append(third_friday.strftime('%Y-%m-%d'))
    
    return sorted(expiries)

AVAILABLE_MATURITIES = generate_quarterly_expiries()

# Cache directory
CACHE_DIR = 'futures_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables for caching status
cache_status = {'status': 'idle', 'progress': 0, 'total': 0, 'current': ''}
cache_lock = threading.Lock()

# Placeholder for your data function
def get_futures_data(start_date, indices, maturities):
    """
    Replace this function with your actual data retrieval function.
    This is a placeholder that generates sample data.
    
    Args:
        start_date: Start date for data retrieval
        indices: List of index names
        maturities: List of maturity dates
    
    Returns:
        DataFrame with columns: Date, Index, Expiry, Price (or your actual columns)
    """
    # Sample data generation - replace with your actual function
    date_range = pd.date_range(start=start_date, end=date.today(), freq='D')
    
    data = []
    for idx in indices:
        for maturity in maturities:
            for dt in date_range:
                # Generate sample price data
                base_price = 100 + hash(f"{idx}_{maturity}") % 50
                price = base_price + (hash(f"{dt}_{idx}_{maturity}") % 20 - 10)
                
                data.append({
                    'Date': dt,
                    'Index': idx,
                    'Expiry': maturity,
                    'Price': max(price, 10),  # Ensure positive prices
                    'index_expiries': f"{idx}_{maturity}"
                })
    
    df = pd.DataFrame(data)
    
    # Simulate processing time
    time.sleep(0.5)  # Remove this in your actual implementation
    
    return df

def get_cache_filename(start_date, indices, maturities):
    """Generate a unique cache filename based on parameters."""
    indices_str = '_'.join(sorted(indices))
    maturities_str = '_'.join(sorted(maturities))
    return f"cache_{start_date}_{hash(indices_str + maturities_str)}.pkl"

def load_from_cache(start_date, indices, maturities):
    """Load data from cache if available."""
    cache_file = os.path.join(CACHE_DIR, get_cache_filename(start_date, indices, maturities))
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_to_cache(data, start_date, indices, maturities):
    """Save data to cache."""
    cache_file = os.path.join(CACHE_DIR, get_cache_filename(start_date, indices, maturities))
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")

def cache_all_data():
    """Cache data for all combinations of indices and maturities."""
    global cache_status
    
    with cache_lock:
        cache_status['status'] = 'running'
        cache_status['progress'] = 0
    
    start_date = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    total_combinations = len(AVAILABLE_INDICES) * len(AVAILABLE_MATURITIES)
    
    with cache_lock:
        cache_status['total'] = total_combinations
    
    count = 0
    for idx in AVAILABLE_INDICES:
        for maturity in AVAILABLE_MATURITIES:
            with cache_lock:
                cache_status['current'] = f"{idx} - {maturity}"
            
            # Check if already cached
            if load_from_cache(start_date, [idx], [maturity]) is None:
                data = get_futures_data(start_date, [idx], [maturity])
                save_to_cache(data, start_date, [idx], [maturity])
            
            count += 1
            with cache_lock:
                cache_status['progress'] = count
    
    with cache_lock:
        cache_status['status'] = 'completed'
        cache_status['current'] = 'All data cached successfully!'

# App layout
app.layout = html.Div([
    html.H1("Futures Index Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Controls section
    html.Div([
        html.Div([
            html.Label("Start Date:", style={'fontWeight': 'bold'}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date=date.today() - timedelta(days=30),
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label("Select Indices:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='indices-dropdown',
                options=[{'label': idx, 'value': idx} for idx in AVAILABLE_INDICES],
                value=[AVAILABLE_INDICES[0]],
                multi=True,
                placeholder="Select indices..."
            ),
        ], style={'width': '33%', 'display': 'inline-block', 'marginRight': '3%'}),
        
        html.Div([
            html.Label("Select Maturities:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='maturities-dropdown',
                options=[{'label': mat, 'value': mat} for mat in AVAILABLE_MATURITIES],
                value=[AVAILABLE_MATURITIES[0]],
                multi=True,
                placeholder="Select maturities..."
            ),
        ], style={'width': '31%', 'display': 'inline-block'}),
    ], style={'marginBottom': 20}),
    
    # Action buttons
    html.Div([
        html.Button('Update Chart', id='update-button', n_clicks=0, 
                   style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 
                          'padding': '10px 20px', 'marginRight': '10px', 'cursor': 'pointer'}),
        html.Button('Cache All Data', id='cache-button', n_clicks=0,
                   style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 
                          'padding': '10px 20px', 'cursor': 'pointer'}),
    ], style={'marginBottom': 20}),
    
    # Cache status
    html.Div(id='cache-status', style={'marginBottom': 20}),
    
    # Chart
    dcc.Graph(id='futures-chart', style={'height': '600px'}),
    
    # Data table (optional)
    html.H3("Data Preview", style={'marginTop': 30}),
    html.Div(id='data-table'),
    
    # Interval component for updating cache status
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    
], style={'margin': '20px'})

# Callback for updating cache status
@app.callback(
    Output('cache-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_cache_status(n):
    global cache_status
    
    with cache_lock:
        status = cache_status.copy()
    
    if status['status'] == 'idle':
        return html.P("Cache Status: Ready", style={'color': 'green'})
    elif status['status'] == 'running':
        progress = status['progress']
        total = status['total']
        current = status['current']
        percentage = (progress / total * 100) if total > 0 else 0
        
        return html.Div([
            html.P(f"Caching in progress: {progress}/{total} ({percentage:.1f}%)", 
                   style={'color': 'orange', 'margin': '5px 0'}),
            html.P(f"Current: {current}", style={'fontSize': '12px', 'color': 'gray', 'margin': '0'}),
            html.Div(
                style={
                    'width': '100%', 'backgroundColor': '#f0f0f0', 'borderRadius': '10px',
                    'height': '20px', 'marginTop': '5px'
                },
                children=[
                    html.Div(
                        style={
                            'width': f'{percentage}%', 'backgroundColor': '#28a745',
                            'height': '100%', 'borderRadius': '10px', 'transition': 'width 0.3s'
                        }
                    )
                ]
            )
        ])
    elif status['status'] == 'completed':
        return html.P("Cache Status: All data cached successfully!", style={'color': 'green'})
    else:
        return html.P("Cache Status: Unknown", style={'color': 'red'})

# Callback for caching all data
@app.callback(
    Output('cache-button', 'children'),
    Input('cache-button', 'n_clicks'),
    prevent_initial_call=True
)
def start_caching(n_clicks):
    if n_clicks > 0:
        # Start caching in a separate thread
        thread = threading.Thread(target=cache_all_data)
        thread.daemon = True
        thread.start()
        return 'Caching...'
    return 'Cache All Data'

# Main callback for updating the chart
@app.callback(
    [Output('futures-chart', 'figure'),
     Output('data-table', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('start-date-picker', 'date'),
     State('indices-dropdown', 'value'),
     State('maturities-dropdown', 'value')],
    prevent_initial_call=True
)
def update_chart(n_clicks, start_date, selected_indices, selected_maturities):
    if not selected_indices or not selected_maturities:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Please select at least one index and one maturity")
        return empty_fig, html.P("No data to display")
    
    # Try to load from cache first
    df = load_from_cache(start_date, selected_indices, selected_maturities)
    
    if df is None:
        # Generate new data
        df = get_futures_data(start_date, selected_indices, selected_maturities)
        # Save to cache
        save_to_cache(df, start_date, selected_indices, selected_maturities)
        cache_source = "Generated (now cached)"
    else:
        cache_source = "Loaded from cache"
    
    # Create the plot
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    
    for idx in selected_indices:
        for maturity in selected_maturities:
            subset = df[(df['Index'] == idx) & (df['Expiry'] == maturity)]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset['Date'],
                    y=subset['Price'],
                    mode='lines',
                    name=f'{idx} - {maturity}',
                    line=dict(color=colors[color_idx % len(colors)]),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Date: %{x}<br>' +
                                  'Price: %{y:.2f}<br>' +
                                  '<extra></extra>'
                ))
                color_idx += 1
    
    fig.update_layout(
        title=f'Futures Price Data - {cache_source}',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        margin=dict(r=150)
    )
    
    # Create data table preview
    preview_df = df.head(10)
    table = dash_table.DataTable(
        data=preview_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in preview_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    )
    
    return fig, table

if __name__ == '__main__':
    app.run_server(debug=True)
