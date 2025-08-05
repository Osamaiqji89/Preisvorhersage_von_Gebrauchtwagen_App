"""
Gebrauchtwagen-Preisvorhersage mit Machine Learning

Dieses Modul implementiert verschiedene ML-Algorithmen zur Vorhersage von Gebrauchtwagenpreisen
basierend auf Fahrzeugmerkmalen wie Marke, Modell, Jahr, Laufleistung, etc.

Forschungsfragen:
1. Welche ML-Algorithmen eignen sich am besten für die Preisvorhersage?
2. Welche Fahrzeugmerkmale haben den größten Einfluss auf den Preis?
3. Wie genau können Preise mit den entwickelten Modellen vorhergesagt werden?
4. Welche praktischen Implikationen ergeben sich für den Gebrauchtwagenmarkt?
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning Imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')

class CarPricePredictor:
    """
    Hauptklasse für die Gebrauchtwagen-Preisvorhersage
    """
    
    def __init__(self, data_path: str = "Daten"):
        """
        Initialisiert den CarPricePredictor
        
        Args:
            data_path: Pfad zum Datenordner mit CSV-Dateien
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.feature_importance = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """
        Lädt alle CSV-Dateien und kombiniert sie zu einem DataFrame
        
        Returns:
            Kombinierter DataFrame mit allen Fahrzeugdaten
        """
        print("Lade Daten aus CSV-Dateien...")
        
        # Alle CSV-Dateien im Datenordner finden
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        # Unclean-Dateien ausschließen
        csv_files = [f for f in csv_files if "unclean" not in f.lower()]
        
        dataframes = []
        
        for file in csv_files:
            # Markenname aus Dateiname extrahieren
            brand = os.path.basename(file).replace('.csv', '')
            
            try:
                df = pd.read_csv(file)
                df['brand'] = brand  # Marke als neue Spalte hinzufügen
                dataframes.append(df)
                print(f"Geladen: {brand} ({len(df)} Datensätze)")
            except Exception as e:
                print(f"Fehler beim Laden von {file}: {e}")
        
        # Alle DataFrames kombinieren
        self.data = pd.concat(dataframes, ignore_index=True)
        print(f"\nGesamtdatensatz: {len(self.data)} Datensätze aus {len(dataframes)} Marken")
        
        return self.data
    
    def explore_data(self) -> Dict[str, Any]:
        """
        Führt eine explorative Datenanalyse durch
        
        Returns:
            Dictionary mit Analyseergebnissen
        """
        print("\n=== EXPLORATIVE DATENANALYSE ===")
        
        # Grundlegende Informationen
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Fehlende Werte
        missing_values = self.data.isnull().sum()
        print(f"\nFehlende Werte:\n{missing_values}")
        
        # Datentypen
        print(f"\nDatentypen:\n{self.data.dtypes}")
        
        # Statistische Zusammenfassung
        print(f"\nStatistische Zusammenfassung:")
        print(self.data.describe())
        
        # Verteilung der Marken
        brand_counts = self.data['brand'].value_counts()
        print(f"\nVerteilung der Marken:\n{brand_counts}")
        
        # Preisstatistiken
        print(f"\nPreisstatistiken:")
        print(f"Durchschnittspreis: £{self.data['price'].mean():.2f}")
        print(f"Medianpreis: £{self.data['price'].median():.2f}")
        print(f"Min/Max Preis: £{self.data['price'].min():.2f} / £{self.data['price'].max():.2f}")
        
        return {
            'shape': self.data.shape,
            'missing_values': missing_values.to_dict(),
            'brand_counts': brand_counts.to_dict(),
            'price_stats': self.data['price'].describe().to_dict()
        }
    
    def clean_data(self) -> pd.DataFrame:
        """
        Bereinigt die Daten und bereitet sie für ML vor
        
        Returns:
            Bereinigter DataFrame
        """
        print("\n=== DATENBEREINIGUNG ===")
        
        # Kopie erstellen
        df = self.data.copy()
        
        # Spalten mit führenden/nachgestellten Leerzeichen bereinigen
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Fehlende Werte behandeln
        print(f"Fehlende Werte vor Bereinigung: {df.isnull().sum().sum()}")
        
        # Numerische Spalten: Median-Imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Kategorische Spalten: Mode-Imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_value, inplace=True)
        
        # Outlier-Erkennung für Preis (IQR-Methode)
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(df)
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        outliers_removed = outliers_before - len(df)
        
        print(f"Outlier entfernt: {outliers_removed}")
        print(f"Datensätze nach Bereinigung: {len(df)}")
        
        self.data = df
        return df
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Bereitet Features für Machine Learning vor
        
        Returns:
            Tuple von (Features, Target)
        """
        print("\n=== FEATURE-ENGINEERING ===")
        
        df = self.data.copy()
        
        # Target-Variable definieren
        y = df['price']
        
        # Features-DataFrame erstellen
        X = df.drop(['price'], axis=1)
        
        # Kategorische Variablen enkodieren
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Feature-Engineering: Neue Features erstellen
        if 'year' in X.columns:
            current_year = 2024
            X['age'] = current_year - X['year']
        
        if 'mileage' in X.columns and 'age' in X.columns:
            X['mileage_per_year'] = X['mileage'] / (X['age'] + 1)  # +1 um Division durch 0 zu vermeiden
        
        if 'engineSize' in X.columns:
            if 'mpg' in X.columns:
                # Vermeidung von Division durch 0 bei engineSize
                X['power_efficiency'] = np.where(X['engineSize'] > 0, 
                                                X['mpg'] / X['engineSize'], 
                                                X['mpg'])
            else:
                X['power_efficiency'] = X['engineSize']
        
        # Unendliche und NaN Werte durch Median ersetzen
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any() or np.isinf(X[col]).any():
                median_val = X[col].replace([np.inf, -np.inf], np.nan).median()
                X[col] = X[col].replace([np.inf, -np.inf, np.nan], median_val)
        
        print(f"Features vorbereitet: {list(X.columns)}")
        print(f"Feature-Matrix Shape: {X.shape}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """
        Teilt Daten in Training- und Testsets auf
        
        Args:
            X: Feature-Matrix
            y: Target-Variable
            test_size: Anteil der Testdaten
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Skalierung der Features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nDatenaufteilung:")
        print(f"Training: {self.X_train.shape[0]} Samples")
        print(f"Test: {self.X_test.shape[0]} Samples")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Trainiert verschiedene ML-Modelle
        
        Returns:
            Dictionary mit trainierten Modellen und Ergebnissen
        """
        print("\n=== MODELL-TRAINING ===")
        
        # Definition der Modelle
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=6)
        }
        
        # Training und Evaluation
        for name, model in self.models.items():
            print(f"\nTrainiere {name}...")
            
            # Für lineare Modelle: skalierte Daten verwenden
            if 'Regression' in name:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Metriken berechnen
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-Validation Score
            if 'Regression' in name:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            
            self.model_results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: £{rmse:.2f}")
            print(f"Cross-Validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return self.model_results
    
    def analyze_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Analysiert die Feature-Wichtigkeit für baumbasierte Modelle
        
        Returns:
            Dictionary mit Feature-Wichtigkeiten
        """
        print("\n=== FEATURE-WICHTIGKEIT ===")
        
        tree_models = ['Decision Tree', 'Random Forest', 'XGBoost']
        
        for model_name in tree_models:
            if model_name in self.model_results:
                model = self.model_results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.X_train.columns
                    
                    # Sortierung nach Wichtigkeit
                    indices = np.argsort(importances)[::-1]
                    
                    self.feature_importance[model_name] = {
                        'importances': importances,
                        'feature_names': feature_names,
                        'sorted_indices': indices
                    }
                    
                    print(f"\n{model_name} - Top 10 Features:")
                    for i in range(min(10, len(indices))):
                        idx = indices[i]
                        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return self.feature_importance
    
    def create_visualizations(self) -> None:
        """
        Erstellt verschiedene Visualisierungen
        """
        print("\n=== VISUALISIERUNGEN ERSTELLEN ===")
        
        # 1. Modell-Vergleich
        self.plot_model_comparison()
        
        # 2. Feature-Wichtigkeit
        self.plot_feature_importance()
        
        # 3. Vorhersage vs. Tatsächliche Werte
        self.plot_predictions()
        
        # 4. Datenverteilungen
        self.plot_data_distributions()
    
    def plot_model_comparison(self) -> None:
        """
        Plottet Vergleich der Modell-Performance
        """
        models = list(self.model_results.keys())
        r2_scores = [self.model_results[model]['r2'] for model in models]
        rmse_scores = [self.model_results[model]['rmse'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² Scores
        bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Modell-Vergleich: R² Score')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Werte auf Balken anzeigen
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE Scores
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Modell-Vergleich: RMSE')
        ax2.set_ylabel('RMSE (£)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Werte auf Balken anzeigen
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{score:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self) -> None:
        """
        Plottet Feature-Wichtigkeit für Random Forest
        """
        if 'Random Forest' in self.feature_importance:
            rf_importance = self.feature_importance['Random Forest']
            importances = rf_importance['importances']
            feature_names = rf_importance['feature_names']
            indices = rf_importance['sorted_indices']
            
            plt.figure(figsize=(12, 8))
            top_n = min(15, len(indices))
            top_indices = indices[:top_n]
            
            plt.barh(range(top_n), importances[top_indices], color='forestgreen', alpha=0.7)
            plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest - Feature Wichtigkeit (Top 15)')
            plt.grid(True, alpha=0.3)
            
            # Werte anzeigen
            for i, (idx, imp) in enumerate(zip(top_indices, importances[top_indices])):
                plt.text(imp + 0.001, i, f'{imp:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_predictions(self) -> None:
        """
        Plottet Vorhersagen vs. tatsächliche Werte für beste Modelle
        """
        best_models = ['Random Forest', 'XGBoost']
        
        fig, axes = plt.subplots(1, len(best_models), figsize=(15, 6))
        if len(best_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(best_models):
            if model_name in self.model_results:
                y_pred = self.model_results[model_name]['predictions']
                r2 = self.model_results[model_name]['r2']
                
                axes[i].scatter(self.y_test, y_pred, alpha=0.5, s=20)
                axes[i].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', linewidth=2)
                axes[i].set_xlabel('Tatsächliche Preise (£)')
                axes[i].set_ylabel('Vorhergesagte Preise (£)')
                axes[i].set_title(f'{model_name}\nR² = {r2:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_data_distributions(self) -> None:
        """
        Plottet wichtige Datenverteilungen
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Preisverteilung
        axes[0,0].hist(self.data['price'], bins=50, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Preisverteilung')
        axes[0,0].set_xlabel('Preis (£)')
        axes[0,0].set_ylabel('Häufigkeit')
        axes[0,0].grid(True, alpha=0.3)
        
        # Markenverteilung
        brand_counts = self.data['brand'].value_counts().head(10)
        axes[0,1].bar(brand_counts.index, brand_counts.values, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Top 10 Marken nach Anzahl')
        axes[0,1].set_ylabel('Anzahl Fahrzeuge')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Alter vs. Preis
        if 'year' in self.data.columns:
            current_year = 2024
            age = current_year - self.data['year']
            axes[1,0].scatter(age, self.data['price'], alpha=0.5, s=10)
            axes[1,0].set_xlabel('Fahrzeugalter (Jahre)')
            axes[1,0].set_ylabel('Preis (£)')
            axes[1,0].set_title('Fahrzeugalter vs. Preis')
            axes[1,0].grid(True, alpha=0.3)
        
        # Laufleistung vs. Preis
        if 'mileage' in self.data.columns:
            axes[1,1].scatter(self.data['mileage'], self.data['price'], alpha=0.5, s=10)
            axes[1,1].set_xlabel('Laufleistung (Meilen)')
            axes[1,1].set_ylabel('Preis (£)')
            axes[1,1].set_title('Laufleistung vs. Preis')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_price(self, car_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Vorhersage des Preises für ein einzelnes Fahrzeug
        
        Args:
            car_features: Dictionary mit Fahrzeugmerkmalen
            
        Returns:
            Dictionary mit Vorhersagen aller Modelle
        """
        # Feature-Vektor erstellen
        feature_vector = pd.DataFrame([car_features])
        
        # Kategorische Variablen enkodieren
        for col in feature_vector.columns:
            if col in self.label_encoders and feature_vector[col].dtype == 'object':
                try:
                    feature_vector[col] = self.label_encoders[col].transform(feature_vector[col])
                except ValueError:
                    # Neuer Wert, verwende häufigsten Wert
                    feature_vector[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])
        
        # Sicherstellen, dass alle erwarteten Features vorhanden sind
        for col in self.X_train.columns:
            if col not in feature_vector.columns:
                feature_vector[col] = 0  # Default-Wert
        
        # Reihenfolge der Features anpassen
        feature_vector = feature_vector[self.X_train.columns]
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if 'Regression' in name:
                    # Skalierte Features für lineare Modelle
                    scaled_features = self.scaler.transform(feature_vector)
                    pred = model.predict(scaled_features)[0]
                else:
                    pred = model.predict(feature_vector)[0]
                
                predictions[name] = max(0, pred)  # Negative Preise vermeiden
            except Exception as e:
                print(f"Fehler bei Vorhersage mit {name}: {e}")
                predictions[name] = 0
        
        return predictions
    
    def generate_report(self) -> str:
        """
        Generiert einen umfassenden Analysebericht
        
        Returns:
            Formatted report string
        """
        report = """
==========================================================
         GEBRAUCHTWAGEN-PREISVORHERSAGE ANALYSEBERICHT
==========================================================

FORSCHUNGSFRAGEN & ANTWORTEN:

1. Welche ML-Algorithmen eignen sich am besten für die Preisvorhersage?
"""
        
        # Beste Modelle identifizieren
        sorted_models = sorted(self.model_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        report += f"\nRANKING DER MODELLE (nach R² Score):\n"
        for i, (name, results) in enumerate(sorted_models):
            report += f"{i+1}. {name}: R² = {results['r2']:.4f}, RMSE = £{results['rmse']:.2f}\n"
        
        best_model = sorted_models[0]
        report += f"\n✓ BESTE PERFORMANCE: {best_model[0]} mit R² = {best_model[1]['r2']:.4f}\n"
        
        report += """
2. Welche Fahrzeugmerkmale haben den größten Einfluss auf den Preis?
"""
        
        if 'Random Forest' in self.feature_importance:
            rf_importance = self.feature_importance['Random Forest']
            importances = rf_importance['importances']
            feature_names = rf_importance['feature_names']
            indices = rf_importance['sorted_indices']
            
            report += "\nTOP 10 WICHTIGSTE FEATURES (Random Forest):\n"
            for i in range(min(10, len(indices))):
                idx = indices[i]
                report += f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}\n"
        
        report += """
3. Wie genau können Preise mit den entwickelten Modellen vorhergesagt werden?
"""
        
        report += f"\nGENAUIGKEITS-METRIKEN des besten Modells ({best_model[0]}):\n"
        report += f"- R² Score: {best_model[1]['r2']:.4f} (Erklärt {best_model[1]['r2']*100:.1f}% der Varianz)\n"
        report += f"- RMSE: £{best_model[1]['rmse']:.2f}\n"
        report += f"- MAE: £{best_model[1]['mae']:.2f}\n"
        report += f"- Cross-Validation R²: {best_model[1]['cv_mean']:.4f} (±{best_model[1]['cv_std']:.4f})\n"
        
        report += """
4. Welche praktischen Implikationen ergeben sich für den Gebrauchtwagenmarkt?
"""
        
        report += f"""
PRAKTISCHE IMPLIKATIONEN:

• MARKTBEWERTUNG: Mit einer Genauigkeit von {best_model[1]['r2']*100:.1f}% können realistische 
  Preisschätzungen für Gebrauchtwagen erstellt werden.

• PREISFAKTOREN: Die wichtigsten Preisfaktoren sind primär fahrzeugspezifische 
  Merkmale, was zeigt, dass traditionelle Bewertungskriterien weiterhin relevant sind.

• MODELL-EMPFEHLUNG: {best_model[0]} liefert die beste Balance zwischen 
  Genauigkeit und Interpretierbarkeit.

• FEHLERBEREICH: Durchschnittlicher Vorhersagefehler von ±£{best_model[1]['rmse']:.0f} 
  ist für praktische Anwendungen akzeptabel.

DATASET-STATISTIKEN:
- Gesamtanzahl Fahrzeuge: {len(self.data):,}
- Durchschnittspreis: £{self.data['price'].mean():.2f}
- Preisspanne: £{self.data['price'].min():.0f} - £{self.data['price'].max():,.0f}
- Anzahl Marken: {self.data['brand'].nunique()}
"""
        
        return report
    
    def run_complete_analysis(self) -> None:
        """
        Führt die komplette Analyse durch
        """
        print("🚗 GEBRAUCHTWAGEN-PREISVORHERSAGE GESTARTET 🚗")
        print("=" * 60)
        
        # 1. Daten laden
        self.load_data()
        
        # 2. Explorative Datenanalyse
        self.explore_data()
        
        # 3. Datenbereinigung
        self.clean_data()
        
        # 4. Feature-Engineering
        X, y = self.prepare_features()
        
        # 5. Daten aufteilen
        self.split_data(X, y)
        
        # 6. Modelle trainieren
        self.train_models()
        
        # 7. Feature-Wichtigkeit analysieren
        self.analyze_feature_importance()
        
        # 8. Visualisierungen erstellen
        self.create_visualizations()
        
        # 9. Bericht generieren
        report = self.generate_report()
        print(report)
        
        # Bericht speichern
        with open('analysebericht.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n✅ ANALYSE ABGESCHLOSSEN!")
        print("📊 Visualisierungen gespeichert als PNG-Dateien")
        print("📄 Detaillierter Bericht gespeichert als 'analysebericht.txt'")

if __name__ == "__main__":
    # Hauptanalyse ausführen
    predictor = CarPricePredictor()
    predictor.run_complete_analysis()
    
    # Beispiel für Einzelvorhersage
    print("\n" + "="*60)
    print("BEISPIEL-VORHERSAGE")
    print("="*60)
    
    example_car = {
        'brand': 'bmw',
        'model': '3 Series',
        'year': 2018,
        'transmission': 'Automatic',
        'mileage': 25000,
        'fuelType': 'Petrol',
        'tax': 145,
        'mpg': 45.0,
        'engineSize': 2.0
    }
    
    predictions = predictor.predict_price(example_car)
    print(f"\nVorhersagen für: {example_car}")
    print("-" * 40)
    for model, price in predictions.items():
        print(f"{model}: £{price:.2f}")
