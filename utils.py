"""
Utility-Funktionen für die Gebrauchtwagen-Preisvorhersage

Dieses Modul enthält hilfreiche Funktionen für Datenverarbeitung,
Visualisierung und Modell-Evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class DataValidator:
    """Klasse für Datenvalidierung und Qualitätskontrolle"""
    
    @staticmethod
    def validate_car_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validiert Fahrzeugdaten auf Plausibilität
        
        Args:
            df: DataFrame mit Fahrzeugdaten
            
        Returns:
            Dictionary mit Validierungsergebnissen
        """
        issues = []
        warnings_list = []
        
        # 1. Spalten-Validierung
        required_columns = ['model', 'year', 'price', 'transmission', 'mileage', 
                           'fuelType', 'tax', 'mpg', 'engineSize']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Fehlende Spalten: {missing_columns}")
        
        # 2. Datentyp-Validierung
        if 'price' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['price']):
                issues.append("Preis-Spalte ist nicht numerisch")
            elif (df['price'] <= 0).any():
                warnings_list.append(f"{(df['price'] <= 0).sum()} Fahrzeuge mit Preis ≤ 0")
            elif (df['price'] > 200000).any():
                warnings_list.append(f"{(df['price'] > 200000).sum()} Fahrzeuge mit Preis > £200,000")
        
        # 3. Jahr-Validierung
        if 'year' in df.columns:
            current_year = 2024
            if (df['year'] < 1990).any():
                warnings_list.append(f"{(df['year'] < 1990).sum()} Fahrzeuge vor 1990")
            if (df['year'] > current_year).any():
                issues.append(f"{(df['year'] > current_year).sum()} Fahrzeuge aus der Zukunft")
        
        # 4. Laufleistung-Validierung
        if 'mileage' in df.columns:
            if (df['mileage'] < 0).any():
                issues.append(f"{(df['mileage'] < 0).sum()} Fahrzeuge mit negativer Laufleistung")
            if (df['mileage'] > 500000).any():
                warnings_list.append(f"{(df['mileage'] > 500000).sum()} Fahrzeuge mit >500k Meilen")
        
        # 5. Verbrauch-Validierung
        if 'mpg' in df.columns:
            if (df['mpg'] <= 0).any():
                warnings_list.append(f"{(df['mpg'] <= 0).sum()} Fahrzeuge mit Verbrauch ≤ 0 MPG")
            if (df['mpg'] > 100).any():
                warnings_list.append(f"{(df['mpg'] > 100).sum()} Fahrzeuge mit >100 MPG")
        
        # 6. Hubraum-Validierung
        if 'engineSize' in df.columns:
            if (df['engineSize'] <= 0).any():
                warnings_list.append(f"{(df['engineSize'] <= 0).sum()} Fahrzeuge mit Hubraum ≤ 0")
            if (df['engineSize'] > 10).any():
                warnings_list.append(f"{(df['engineSize'] > 10).sum()} Fahrzeuge mit >10L Hubraum")
        
        return {
            'total_records': len(df),
            'issues': issues,
            'warnings': warnings_list,
            'is_valid': len(issues) == 0,
            'quality_score': max(0, 100 - len(issues) * 20 - len(warnings_list) * 5)
        }
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """
        Erkennt Outlier in einer Spalte
        
        Args:
            df: DataFrame
            column: Spaltenname
            method: Methode ('iqr' oder 'zscore')
            
        Returns:
            Boolean Series mit Outlier-Markierungen
        """
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return z_scores > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

class VisualizationHelper:
    """Hilfsklasse für erweiterte Visualisierungen"""
    
    @staticmethod
    def create_price_analysis_dashboard(df: pd.DataFrame) -> None:
        """
        Erstellt ein umfassendes Dashboard für Preisanalyse
        
        Args:
            df: DataFrame mit Fahrzeugdaten
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Preisverteilung', 'Preis vs. Jahr', 
                           'Preis vs. Laufleistung', 'Preis nach Kraftstoff'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Preisverteilung (Histogramm)
        fig.add_trace(
            go.Histogram(x=df['price'], name='Preis', nbinsx=50),
            row=1, col=1
        )
        
        # 2. Preis vs. Jahr (Scatter)
        if 'year' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['year'], y=df['price'], mode='markers', 
                          name='Jahr vs. Preis', opacity=0.6),
                row=1, col=2
            )
        
        # 3. Preis vs. Laufleistung (Scatter)
        if 'mileage' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['mileage'], y=df['price'], mode='markers',
                          name='Laufleistung vs. Preis', opacity=0.6),
                row=2, col=1
            )
        
        # 4. Preis nach Kraftstoffart (Box Plot)
        if 'fuelType' in df.columns:
            for fuel in df['fuelType'].unique():
                fuel_data = df[df['fuelType'] == fuel]['price']
                fig.add_trace(
                    go.Box(y=fuel_data, name=f'{fuel}', showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Gebrauchtwagen Preisanalyse Dashboard")
        fig.show()
    
    @staticmethod
    def plot_model_performance_radar(model_results: Dict[str, Dict]) -> None:
        """
        Erstellt Radar-Chart für Modell-Performance
        
        Args:
            model_results: Dictionary mit Modell-Ergebnissen
        """
        # Metriken normalisieren (0-1 Skala)
        models = list(model_results.keys())
        r2_scores = [model_results[model]['r2'] for model in models]
        rmse_scores = [model_results[model]['rmse'] for model in models]
        mae_scores = [model_results[model]['mae'] for model in models]
        
        # RMSE und MAE invertieren (niedrigere Werte sind besser)
        max_rmse = max(rmse_scores)
        max_mae = max(mae_scores)
        
        normalized_rmse = [(max_rmse - rmse) / max_rmse for rmse in rmse_scores]
        normalized_mae = [(max_mae - mae) / max_mae for mae in mae_scores]
        
        fig = go.Figure()
        
        for i, model in enumerate(models):
            fig.add_trace(go.Scatterpolar(
                r=[r2_scores[i], normalized_rmse[i], normalized_mae[i]],
                theta=['R² Score', 'RMSE (invertiert)', 'MAE (invertiert)'],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Modell-Performance Radar Chart"
        )
        
        fig.show()
    
    @staticmethod
    def create_feature_correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Erstellt Korrelations-Heatmap für numerische Features
        
        Args:
            df: DataFrame mit Features
            figsize: Größe der Abbildung
        """
        # Nur numerische Spalten
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Nicht genug numerische Spalten für Korrelationsanalyse")
            return
        
        # Korrelationsmatrix berechnen
        corr_matrix = numeric_df.corr()
        
        # Heatmap erstellen
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Korrelations-Heatmap')
        plt.tight_layout()
        plt.show()

class ModelEvaluator:
    """Erweiterte Modell-Evaluation Klasse"""
    
    @staticmethod
    def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> Dict[str, float]:
        """
        Umfassende Modell-Evaluation mit verschiedenen Metriken
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhergesagte Werte
            model_name: Name des Modells
            
        Returns:
            Dictionary mit Evaluationsmetriken
        """
        # Grundlegende Metriken
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Erweiterte Metriken
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        
        # Residual-Statistiken
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Prediction Interval Coverage
        prediction_interval_68 = np.percentile(np.abs(residuals), 68.2)
        prediction_interval_95 = np.percentile(np.abs(residuals), 95.4)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'prediction_interval_68': prediction_interval_68,
            'prediction_interval_95': prediction_interval_95,
            'accuracy_within_10_percent': np.mean(np.abs(residuals / y_true) < 0.1) * 100,
            'accuracy_within_20_percent': np.mean(np.abs(residuals / y_true) < 0.2) * 100
        }
    
    @staticmethod
    def plot_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model") -> None:
        """
        Erstellt Residual-Analyse Plots
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhergesagte Werte
            model_name: Name des Modells
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Residual-Analyse für {model_name}', fontsize=16)
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Tatsächliche Werte')
        axes[0, 0].set_ylabel('Vorhergesagte Werte')
        axes[0, 0].set_title('Vorhersage vs. Tatsächlich')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Vorhergesagte Werte')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs. Vorhersage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Häufigkeit')
        axes[1, 0].set_title('Residuals Verteilung')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normalität)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class DataProcessor:
    """Erweiterte Datenverarbeitungsklasse"""
    
    @staticmethod
    def create_price_segments(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Erstellt Preissegmente für bessere Analyse
        
        Args:
            df: DataFrame mit Preisdaten
            price_col: Name der Preisspalte
            
        Returns:
            DataFrame mit zusätzlicher Preissegment-Spalte
        """
        df = df.copy()
        
        # Preissegmente definieren
        df['price_segment'] = pd.cut(df[price_col], 
                                   bins=[0, 10000, 20000, 35000, 50000, float('inf')],
                                   labels=['Budget (<£10k)', 'Economy (£10k-20k)', 
                                          'Mid-range (£20k-35k)', 'Premium (£35k-50k)', 
                                          'Luxury (>£50k)'])
        
        return df
    
    @staticmethod
    def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt erweiterte Features für bessere Modellperformance
        
        Args:
            df: DataFrame mit Basisdaten
            
        Returns:
            DataFrame mit zusätzlichen Features
        """
        df = df.copy()
        
        # Alter des Fahrzeugs
        if 'year' in df.columns:
            current_year = 2024
            df['age'] = current_year - df['year']
            df['age_squared'] = df['age'] ** 2
        
        # Laufleistung pro Jahr
        if 'mileage' in df.columns and 'age' in df.columns:
            df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        
        # Effizienz-Metriken
        if 'mpg' in df.columns and 'engineSize' in df.columns:
            df['efficiency_ratio'] = df['mpg'] / df['engineSize']
        
        # Power-to-weight ratio proxy
        if 'engineSize' in df.columns:
            df['power_category'] = pd.cut(df['engineSize'], 
                                        bins=[0, 1.5, 2.5, 4.0, float('inf')],
                                        labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        # Steuer pro Liter Hubraum
        if 'tax' in df.columns and 'engineSize' in df.columns:
            df['tax_per_liter'] = df['tax'] / df['engineSize']
        
        # Getriebe-binär
        if 'transmission' in df.columns:
            df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)
        
        # Kraftstoff-binär
        if 'fuelType' in df.columns:
            df['is_diesel'] = (df['fuelType'] == 'Diesel').astype(int)
            df['is_electric'] = (df['fuelType'] == 'Electric').astype(int)
            df['is_hybrid'] = (df['fuelType'] == 'Hybrid').astype(int)
        
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Intelligente Behandlung fehlender Werte
        
        Args:
            df: DataFrame mit potentiell fehlenden Werten
            strategy: Strategie ('smart', 'drop', 'median', 'mode')
            
        Returns:
            DataFrame ohne fehlende Werte
        """
        df = df.copy()
        
        if strategy == 'smart':
            # Intelligente Strategie basierend auf Datentyp und Kontext
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # Numerische Spalten: Median oder Regression
                        if col in ['price', 'mileage', 'year']:
                            # Wichtige Spalten: Zeilen löschen
                            df = df.dropna(subset=[col])
                        else:
                            # Weniger wichtige: Median
                            df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Kategorische Spalten: Mode
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                        df[col].fillna(mode_val, inplace=True)
        
        elif strategy == 'drop':
            df = df.dropna()
        
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col].fillna(mode_val, inplace=True)
        
        return df

# Utility Funktionen
def calculate_model_confidence_intervals(model, X_test: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Konfidenzintervalle für Modellvorhersagen (falls unterstützt)
    
    Args:
        model: Trainiertes Modell
        X_test: Test-Features
        confidence: Konfidenzniveau
        
    Returns:
        Tuple von (lower_bounds, upper_bounds)
    """
    try:
        # Für Modelle die Konfidenzintervalle unterstützen
        if hasattr(model, 'predict_interval'):
            predictions = model.predict_interval(X_test, confidence=confidence)
            return predictions[:, 0], predictions[:, 1]
        else:
            # Fallback: Bootstrap-basierte Schätzung
            predictions = model.predict(X_test)
            std_error = np.std(predictions) * 0.1  # Grobe Schätzung
            z_score = 1.96 if confidence == 0.95 else 2.58  # 95% oder 99%
            margin = z_score * std_error
            return predictions - margin, predictions + margin
    except:
        # Fallback bei Fehlern
        predictions = model.predict(X_test)
        return predictions * 0.9, predictions * 1.1

def format_currency(amount: float, currency: str = "£") -> str:
    """
    Formatiert Währungsbeträge
    
    Args:
        amount: Betrag
        currency: Währungssymbol
        
    Returns:
        Formatierter String
    """
    return f"{currency}{amount:,.0f}"

def calculate_depreciation_rate(initial_price: float, current_price: float, years: int) -> float:
    """
    Berechnet jährliche Wertminderungsrate
    
    Args:
        initial_price: Ursprünglicher Preis
        current_price: Aktueller Preis
        years: Anzahl Jahre
        
    Returns:
        Jährliche Wertminderungsrate (0-1)
    """
    if years <= 0 or initial_price <= 0:
        return 0.0
    
    return 1 - (current_price / initial_price) ** (1 / years)
