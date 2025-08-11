High US–Japan interest rate differential → BoJ still dovish, Fed holding rates high → carry trades boosting USD/JPY and indirectly USD/CAD.
	•	Stronger USD across the board → Broad USD moves pushing both pairs in the same direction.
	•	Global risk sentiment alignment → Yen weakens when risk appetite is strong, CAD weakens when commodities or global growth outlook drops—both linked to risk cycles.
	•	Oil price weakness → Lower oil prices since February weakened CAD; global slowdown fears also pressured JPY, aligning their moves.
	•	Commodity & macro shocks → Events like China growth concerns hit both CAD (via trade & oil) and JPY (via safe-haven outflows)


BoJ intervention risk → If yen weakness continues, Tokyo could verbally or physically intervene, breaking alignment with CAD.
	•	Oil price rebound → CAD could strengthen if oil rises, diverging from JPY moves.
	•	Fed rate cut speculation → Any dovish pivot from the Fed might weaken USD across pairs, but not necessarily equally.
	•	Canada-specific data shocks → CPI or GDP surprises in Canada could move CAD independently.
	•	Japan domestic shifts → Wage data or inflation surprises could push BoJ toward tightening earlier than expected, decoupling USD/JPY from USD/CAD.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
import warnings
warnings.filterwarnings('ignore')

class FXHedgeBacktest:
    def __init__(self, df, fx_pair='EURUSD', index_name='SPX', initial_units=1000):
        """
        Initialise le backtest de hedge FX
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame avec colonnes: datetime (index), fx_pair, index_name
        fx_pair : str
            Nom de la paire de devises (ex: 'EURUSD')
        index_name : str  
            Nom de l'indice (ex: 'SPX')
        initial_units : float
            Nombre d'unités initiales de l'indice
        """
        self.df = df.copy()
        self.fx_pair = fx_pair
        self.index_name = index_name
        self.initial_units = initial_units
        
        # Vérifier que les colonnes existent
        required_cols = [fx_pair, index_name]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
        
        # S'assurer que l'index est datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index du DataFrame doit être de type DatetimeIndex")
        
        self.results = None
        self.daily_pnl = None
        
    def run_backtest(self, rehedge_frequency_minutes=1, start_hour=16, end_hour=21):
        """
        Exécute le backtest principal
        
        Parameters:
        -----------
        rehedge_frequency_minutes : int
            Fréquence de rebalancement en minutes (1, 30, 60, etc.)
        start_hour : int
            Heure de début du trading (16 pour 16h)
        end_hour : int
            Heure de fin du trading (21 pour 21h)
        """
        # Filtrer les données pour les heures de trading
        df_filtered = self.df.between_time(f'{start_hour:02d}:00', f'{end_hour-1:02d}:59')
        
        # Grouper par date
        daily_results = []
        
        for date, day_data in df_filtered.groupby(df_filtered.index.date):
            if len(day_data) == 0:
                continue
                
            daily_pnl = self._calculate_daily_pnl(
                day_data, 
                rehedge_frequency_minutes,
                start_hour
            )
            
            if daily_pnl is not None:
                daily_results.append({
                    'date': date,
                    'daily_pnl': daily_pnl['total_pnl'],
                    'num_rehedges': daily_pnl['num_rehedges'],
                    'detailed_pnl': daily_pnl
                })
        
        self.results = pd.DataFrame(daily_results)
        self.rehedge_frequency = rehedge_frequency_minutes
        self.trading_hours = (start_hour, end_hour)
        
        # Calculer les statistiques cumulées
        if len(self.results) > 0:
            self.results['cumulative_pnl'] = self.results['daily_pnl'].cumsum()
            
        return self.results
    
    def _calculate_daily_pnl(self, day_data, rehedge_freq, start_hour):
        """
        Calcule le PnL pour une journée donnée
        """
        if len(day_data) == 0:
            return None
            
        # Point de référence: première observation à partir de start_hour
        start_time = time(start_hour, 0)
        reference_point = day_data[day_data.index.time >= start_time].iloc[0:1]
        
        if len(reference_point) == 0:
            return None
            
        # Valeurs de référence à 16h (ou heure de début)
        ref_fx = reference_point[self.fx_pair].iloc[0]
        ref_index = reference_point[self.index_name].iloc[0]
        ref_value_usd = self.initial_units * ref_index
        
        # Sélectionner les points de rebalancement
        rehedge_data = self._get_rehedge_points(day_data, rehedge_freq, start_hour)
        
        if len(rehedge_data) <= 1:
            return {'total_pnl': 0.0, 'num_rehedges': 0, 'pnl_details': []}
        
        pnl_details = []
        total_pnl = 0.0
        current_hedged_amount_eur = ref_value_usd / ref_fx  # Montant initial hedgé en EUR
        
        for i in range(1, len(rehedge_data)):
            current_row = rehedge_data.iloc[i]
            
            # Nouvelle valeur de l'indice en USD
            new_index_value = current_row[self.index_name]
            new_value_usd = self.initial_units * new_index_value
            
            # Nouveau taux FX
            new_fx = current_row[self.fx_pair]
            
            # Calcul du delta en USD
            delta_usd = new_value_usd - ref_value_usd
            
            if abs(delta_usd) > 0.01:  # Seuil minimal pour éviter les micro-transactions
                # Conversion du delta en EUR au nouveau taux
                delta_eur_at_new_rate = delta_usd / new_fx
                
                # Si on avait converti au taux de référence
                delta_eur_at_ref_rate = delta_usd / ref_fx
                
                # PnL FX = différence entre les deux conversions
                fx_pnl_eur = delta_eur_at_ref_rate - delta_eur_at_new_rate
                fx_pnl_usd = fx_pnl_eur * new_fx
                
                total_pnl += fx_pnl_usd
                
                pnl_details.append({
                    'timestamp': current_row.name,
                    'delta_usd': delta_usd,
                    'fx_rate': new_fx,
                    'ref_fx_rate': ref_fx,
                    'fx_pnl_usd': fx_pnl_usd,
                    'cumulative_pnl': total_pnl
                })
                
                # Mettre à jour la valeur de référence pour le prochain calcul
                ref_value_usd = new_value_usd
        
        return {
            'total_pnl': total_pnl,
            'num_rehedges': len(pnl_details),
            'pnl_details': pnl_details
        }
    
    def _get_rehedge_points(self, day_data, frequency_minutes, start_hour):
        """
        Sélectionne les points de rebalancement selon la fréquence
        """
        if frequency_minutes == 1:
            return day_data
        
        # Créer un échantillonnage à la fréquence demandée
        start_time = day_data.index[0].replace(second=0, microsecond=0)
        
        # Ajuster à l'heure de début si nécessaire
        if start_time.hour < start_hour:
            start_time = start_time.replace(hour=start_hour, minute=0)
        
        # Créer les timestamps de rebalancement
        rehedge_times = []
        current_time = start_time
        end_time = day_data.index[-1]
        
        while current_time <= end_time:
            rehedge_times.append(current_time)
            current_time += pd.Timedelta(minutes=frequency_minutes)
        
        # Sélectionner les données les plus proches de ces timestamps
        rehedge_data = []
        for target_time in rehedge_times:
            # Trouver l'observation la plus proche
            time_diffs = abs(day_data.index - target_time)
            closest_idx = time_diffs.idxmin()
            if time_diffs[closest_idx] <= pd.Timedelta(minutes=frequency_minutes/2):
                rehedge_data.append(day_data.loc[closest_idx])
        
        if rehedge_data:
            return pd.DataFrame(rehedge_data)
        else:
            return day_data.iloc[:1]  # Au moins le premier point
    
    def get_daily_stats(self, date=None):
        """
        Obtient les statistiques détaillées pour une date donnée ou globales
        """
        if self.results is None:
            raise ValueError("Veuillez d'abord exécuter le backtest avec run_backtest()")
        
        if date is not None:
            # Statistiques pour une date spécifique
            if isinstance(date, str):
                date = pd.to_datetime(date).date()
            
            day_result = self.results[self.results['date'] == date]
            if len(day_result) == 0:
                print(f"Aucune donnée trouvée pour la date {date}")
                return None
            
            day_data = day_result.iloc[0]
            detailed_pnl = day_data['detailed_pnl']
            
            stats = {
                'date': date,
                'total_pnl': day_data['daily_pnl'],
                'num_rehedges': day_data['num_rehedges'],
                'pnl_details': detailed_pnl['pnl_details'] if 'pnl_details' in detailed_pnl else []
            }
            
            return stats
        else:
            # Statistiques globales
            total_pnl = self.results['daily_pnl'].sum()
            avg_daily_pnl = self.results['daily_pnl'].mean()
            std_daily_pnl = self.results['daily_pnl'].std()
            max_daily_pnl = self.results['daily_pnl'].max()
            min_daily_pnl = self.results['daily_pnl'].min()
            total_rehedges = self.results['num_rehedges'].sum()
            
            stats = {
                'period': f"{self.results['date'].min()} to {self.results['date'].max()}",
                'total_days': len(self.results),
                'total_pnl': total_pnl,
                'avg_daily_pnl': avg_daily_pnl,
                'std_daily_pnl': std_daily_pnl,
                'max_daily_pnl': max_daily_pnl,
                'min_daily_pnl': min_daily_pnl,
                'total_rehedges': total_rehedges,
                'avg_rehedges_per_day': total_rehedges / len(self.results),
                'sharpe_ratio': avg_daily_pnl / std_daily_pnl if std_daily_pnl > 0 else 0,
                'win_rate': (self.results['daily_pnl'] > 0).mean()
            }
            
            return stats
    
    def plot_results(self, date=None, figsize=(15, 10)):
        """
        Affiche les graphiques des résultats
        """
        if self.results is None:
            raise ValueError("Veuillez d'abord exécuter le backtest avec run_backtest()")
        
        if date is not None:
            self._plot_daily_results(date, figsize)
        else:
            self._plot_overall_results(figsize)
    
    def _plot_daily_results(self, date, figsize):
        """
        Affiche les résultats pour une journée spécifique
        """
        stats = self.get_daily_stats(date)
        if stats is None:
            return
        
        if not stats['pnl_details']:
            print(f"Aucun détail de PnL disponible pour {date}")
            return
        
        pnl_df = pd.DataFrame(stats['pnl_details'])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'FX Hedge PnL Analysis - {date}', fontsize=16)
        
        # PnL cumulatif
        axes[0, 0].plot(pnl_df['timestamp'], pnl_df['cumulative_pnl'], 'b-', linewidth=2)
        axes[0, 0].set_title('PnL Cumulatif')
        axes[0, 0].set_xlabel('Temps')
        axes[0, 0].set_ylabel('PnL (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # PnL par trade
        axes[0, 1].bar(range(len(pnl_df)), pnl_df['fx_pnl_usd'], 
                      color=['green' if x > 0 else 'red' for x in pnl_df['fx_pnl_usd']])
        axes[0, 1].set_title('PnL par Rebalancement')
        axes[0, 1].set_xlabel('Numéro de Rebalancement')
        axes[0, 1].set_ylabel('PnL (USD)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Taux FX
        axes[1, 0].plot(pnl_df['timestamp'], pnl_df['fx_rate'], 'orange', linewidth=2)
        axes[1, 0].axhline(y=pnl_df['ref_fx_rate'].iloc[0], color='red', 
                          linestyle='--', label=f'Taux ref: {pnl_df["ref_fx_rate"].iloc[0]:.4f}')
        axes[1, 0].set_title(f'Évolution du taux {self.fx_pair}')
        axes[1, 0].set_xlabel('Temps')
        axes[1, 0].set_ylabel('Taux FX')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Delta USD
        axes[1, 1].plot(pnl_df['timestamp'], pnl_df['delta_usd'], 'purple', linewidth=2)
        axes[1, 1].set_title('Variation de la position (USD)')
        axes[1, 1].set_xlabel('Temps')
        axes[1, 1].set_ylabel('Delta (USD)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques
        print(f"\n=== Statistiques pour {date} ===")
        print(f"PnL total: ${stats['total_pnl']:.2f}")
        print(f"Nombre de rebalancements: {stats['num_rehedges']}")
        print(f"PnL moyen par rebalancement: ${stats['total_pnl']/max(stats['num_rehedges'], 1):.2f}")
    
    def _plot_overall_results(self, figsize):
        """
        Affiche les résultats globaux sur toute la période
        """
        stats = self.get_daily_stats()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'FX Hedge Backtest - Résultats Globaux\n{self.fx_pair} / {self.index_name}', fontsize=16)
        
        # PnL cumulatif
        axes[0, 0].plot(self.results['date'], self.results['cumulative_pnl'], 'b-', linewidth=2)
        axes[0, 0].set_title('PnL Cumulatif')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('PnL Cumulatif (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # PnL journalier
        colors = ['green' if x > 0 else 'red' for x in self.results['daily_pnl']]
        axes[0, 1].bar(self.results['date'], self.results['daily_pnl'], color=colors, alpha=0.7)
        axes[0, 1].set_title('PnL Journalier')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('PnL (USD)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Distribution des PnL journaliers
        axes[1, 0].hist(self.results['daily_pnl'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=self.results['daily_pnl'].mean(), color='red', 
                          linestyle='--', label=f'Moyenne: ${self.results["daily_pnl"].mean():.2f}')
        axes[1, 0].set_title('Distribution des PnL Journaliers')
        axes[1, 0].set_xlabel('PnL (USD)')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Nombre de rebalancements par jour
        axes[1, 1].plot(self.results['date'], self.results['num_rehedges'], 'go-', alpha=0.7)
        axes[1, 1].set_title('Rebalancements par Jour')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Nombre de Rebalancements')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques globales
        print(f"\n=== Statistiques Globales ===")
        print(f"Période: {stats['period']}")
        print(f"Nombre de jours: {stats['total_days']}")
        print(f"PnL total: ${stats['total_pnl']:.2f}")
        print(f"PnL journalier moyen: ${stats['avg_daily_pnl']:.2f}")
        print(f"Volatilité journalière: ${stats['std_daily_pnl']:.2f}")
        print(f"Ratio de Sharpe: {stats['sharpe_ratio']:.3f}")
        print(f"Taux de réussite: {stats['win_rate']:.1%}")
        print(f"PnL max (1 jour): ${stats['max_daily_pnl']:.2f}")
        print(f"PnL min (1 jour): ${stats['min_daily_pnl']:.2f}")
        print(f"Rebalancements totaux: {stats['total_rehedges']}")
        print(f"Rebalancements/jour (moyenne): {stats['avg_rehedges_per_day']:.1f}")

# Exemple d'utilisation
"""
# Charger vos données
# df doit avoir un DatetimeIndex et les colonnes 'EURUSD' et 'SPX'
# df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Initialiser le backtest
backtest = FXHedgeBacktest(df, fx_pair='EURUSD', index_name='SPX', initial_units=1000)

# Exécuter le backtest avec rebalancement toutes les minutes
results = backtest.run_backtest(rehedge_frequency_minutes=1, start_hour=16, end_hour=21)

# Voir les résultats globaux
backtest.plot_results()
stats = backtest.get_daily_stats()

# Analyser une journée spécifique
backtest.plot_results(date='2024-01-15')
daily_stats = backtest.get_daily_stats('2024-01-15')

# Tester différentes fréquences
backtest_30min = FXHedgeBacktest(df, fx_pair='EURUSD', index_name='SPX', initial_units=1000)
results_30min = backtest_30min.run_backtest(rehedge_frequency_minutes=30)
"""






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
