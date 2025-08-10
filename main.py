import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_fx_hedge_backtest(df, 
                          index_col="SPX", 
                          fx_col="EURUSD", 
                          start_hour=16, 
                          end_hour=21, 
                          units_index=1000, 
                          rehedge_minutes=1):
    """
    Backtest PnL FX hedge entre start_hour et end_hour.
    
    df : DataFrame avec colonnes datetime (timezone aware ou naive), index_price et fx_rate
    index_col : nom de la colonne prix de l'index (en USD)
    fx_col : nom de la colonne prix du FX (USD par EUR ou autre)
    units_index : nombre d'unités de l'index détenues
    rehedge_minutes : fréquence en minutes pour rebalancer le hedge
    """
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Filtrer uniquement les heures d'intérêt
    df = df.between_time(f"{start_hour}:00", f"{end_hour}:00")
    
    results = []
    
    for date, day_data in df.groupby(df.index.date):
        day_data = day_data.resample(f"{rehedge_minutes}T").first().dropna()
        
        if day_data.empty:
            continue
        
        # Valeur initiale de l'index en USD à start_hour
        start_price_index = day_data[index_col].iloc[0]
        start_fx = day_data[fx_col].iloc[0]
        
        position_usd = start_price_index * units_index
        hedge_fx_rate = start_fx  # initial hedge FX rate
        
        cumulative_pnl = 0.0
        pnl_series = []
        
        for t in range(1, len(day_data)):
            current_price_index = day_data[index_col].iloc[t]
            current_fx = day_data[fx_col].iloc[t]
            
            # Valeur actuelle de la position index en USD
            new_value_usd = current_price_index * units_index
            
            # Variation de valeur en USD
            delta_usd = new_value_usd - position_usd
            
            # On ajuste le hedge : on "achète" ou "vend" delta_usd au nouveau FX
            # Le coût/gain FX = montant * (1/hedge_rate - 1/current_rate)
            pnl_fx = delta_usd * (1/hedge_fx_rate - 1/current_fx)
            
            cumulative_pnl += pnl_fx
            pnl_series.append((day_data.index[t], cumulative_pnl))
            
            # On remet à jour la position et le hedge FX
            position_usd = new_value_usd
            hedge_fx_rate = current_fx
        
        daily_df = pd.DataFrame(pnl_series, columns=["datetime", "cumulative_pnl"])
        daily_df['date'] = date
        results.append(daily_df)
    
    all_results = pd.concat(results)
    return all_results


def plot_hedge_pnl(results_df, day=None):
    """
    Trace le PnL d'un jour donné ou de toute la période.
    
    results_df : DataFrame retourné par run_fx_hedge_backtest
    day : date (YYYY-MM-DD) ou None pour toute la période
    """
    if day:
        data = results_df[results_df['date'] == pd.to_datetime(day).date()]
        title = f"PnL FX Hedge - {day}"
    else:
        data = results_df
        title = "PnL FX Hedge - Toute période"
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['datetime'], data['cumulative_pnl'], label="Cumulative PnL")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("PnL (base currency)")
    plt.legend()
    plt.grid(True)
    plt.show()
