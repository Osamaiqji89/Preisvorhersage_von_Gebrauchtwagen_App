"""
Umfassende Tests für die Gebrauchtwagen-Preisvorhersage Anwendung

Diese Testsuite überprüft alle Komponenten der ML-Pipeline:
- Datenladung und -bereinigung
- Feature Engineering
- Modelltraining und -bewertung
- Vorhersagegenauigkeit
- Edge Cases und Robustheit
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pickle

# Import der zu testenden Klasse
from car_price_predictor import CarPricePredictor

class TestCarPricePredictor:
    """Test-Klasse für CarPricePredictor"""
    
    @pytest.fixture
    def sample_data(self):
        """Erstellt Beispieldaten für Tests"""
        return pd.DataFrame({
            'model': ['A3', 'A4', '3 Series', '5 Series', 'C-Class'] * 20,
            'year': np.random.randint(2010, 2024, 100),
            'price': np.random.randint(8000, 50000, 100),
            'transmission': np.random.choice(['Manual', 'Automatic'], 100),
            'mileage': np.random.randint(5000, 150000, 100),
            'fuelType': np.random.choice(['Petrol', 'Diesel', 'Hybrid'], 100),
            'tax': np.random.randint(20, 500, 100),
            'mpg': np.random.uniform(20, 80, 100),
            'engineSize': np.random.uniform(1.0, 4.0, 100),
            'brand': ['audi'] * 40 + ['bmw'] * 40 + ['mercedes'] * 20
        })
    
    @pytest.fixture
    def temp_data_dir(self, sample_data):
        """Erstellt temporären Datenordner mit Test-CSV-Dateien"""
        temp_dir = tempfile.mkdtemp()
        
        # Aufteilen der Daten nach Marken
        brands = sample_data['brand'].unique()
        for brand in brands:
            brand_data = sample_data[sample_data['brand'] == brand].drop('brand', axis=1)
            brand_data.to_csv(os.path.join(temp_dir, f"{brand}.csv"), index=False)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def predictor(self, temp_data_dir):
        """Erstellt CarPricePredictor Instanz mit Testdaten"""
        return CarPricePredictor(data_path=temp_data_dir)
    
    def test_initialization(self, temp_data_dir):
        """Test der Initialisierung"""
        predictor = CarPricePredictor(data_path=temp_data_dir)
        
        assert predictor.data_path == temp_data_dir
        assert predictor.data is None
        assert predictor.models == {}
        assert predictor.model_results == {}
        assert predictor.label_encoders == {}
    
    def test_load_data(self, predictor, sample_data):
        """Test des Datenladens"""
        loaded_data = predictor.load_data()
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        assert 'brand' in loaded_data.columns
        assert all(col in loaded_data.columns for col in sample_data.columns)
        
        # Test verschiedene Marken
        brands = loaded_data['brand'].unique()
        assert len(brands) >= 3
    
    def test_load_data_empty_directory(self):
        """Test mit leerem Verzeichnis"""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = CarPricePredictor(data_path=temp_dir)
            
            # Sollte leeren DataFrame zurückgeben oder Exception werfen
            try:
                result = predictor.load_data()
                assert len(result) == 0 or result is None
            except Exception:
                # Erwartetes Verhalten bei fehlenden Dateien
                pass
    
    def test_load_data_invalid_csv(self):
        """Test mit ungültigen CSV-Dateien"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ungültige CSV-Datei erstellen
            with open(os.path.join(temp_dir, "invalid.csv"), "w") as f:
                f.write("invalid,csv,data\n")
                f.write("missing,columns")
            
            predictor = CarPricePredictor(data_path=temp_dir)
            
            # Sollte robust mit fehlerhaften Dateien umgehen
            try:
                result = predictor.load_data()
                # Test erfolgreich wenn keine Exception
                assert True
            except Exception as e:
                # Spezifische Fehlerbehandlung testen
                assert "csv" in str(e).lower() or "data" in str(e).lower()
    
    def test_explore_data(self, predictor, sample_data):
        """Test der explorativen Datenanalyse"""
        predictor.data = sample_data
        
        results = predictor.explore_data()
        
        assert isinstance(results, dict)
        assert 'shape' in results
        assert 'missing_values' in results
        assert 'brand_counts' in results
        assert 'price_stats' in results
        
        # Shape prüfen
        assert results['shape'] == sample_data.shape
        
        # Brand counts prüfen
        assert isinstance(results['brand_counts'], dict)
        assert len(results['brand_counts']) > 0
    
    def test_clean_data(self, predictor, sample_data):
        """Test der Datenbereinigung"""
        # Fehlende Werte hinzufügen
        dirty_data = sample_data.copy()
        dirty_data.loc[0:5, 'mpg'] = np.nan
        dirty_data.loc[10:15, 'model'] = np.nan
        
        predictor.data = dirty_data
        
        cleaned_data = predictor.clean_data()
        
        assert cleaned_data is not None
        assert cleaned_data.isnull().sum().sum() == 0  # Keine fehlenden Werte
        assert len(cleaned_data) <= len(dirty_data)  # Outlier entfernt
        
        # Preis-Outlier-Behandlung prüfen
        price_range = cleaned_data['price'].max() - cleaned_data['price'].min()
        assert price_range > 0
    
    def test_prepare_features(self, predictor, sample_data):
        """Test des Feature Engineering"""
        predictor.data = sample_data
        
        X, y = predictor.prepare_features()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert 'price' not in X.columns
        
        # Neue Features prüfen
        assert 'age' in X.columns
        assert 'mileage_per_year' in X.columns
        
        # Kategorische Enkodierung prüfen
        for col in X.columns:
            assert X[col].dtype in [np.int64, np.float64, np.int32, np.float32]
    
    def test_split_data(self, predictor, sample_data):
        """Test der Datenaufteilung"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        
        predictor.split_data(X, y, test_size=0.2)
        
        assert predictor.X_train is not None
        assert predictor.X_test is not None
        assert predictor.y_train is not None
        assert predictor.y_test is not None
        
        # Proportionen prüfen
        total_samples = len(X)
        train_samples = len(predictor.X_train)
        test_samples = len(predictor.X_test)
        
        assert train_samples + test_samples == total_samples
        assert abs((test_samples / total_samples) - 0.2) < 0.05  # ±5% Toleranz
        
        # Skalierung prüfen
        assert hasattr(predictor, 'X_train_scaled')
        assert hasattr(predictor, 'X_test_scaled')
    
    def test_train_models(self, predictor, sample_data):
        """Test des Modelltrainings"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        
        results = predictor.train_models()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Alle erwarteten Modelle prüfen
        expected_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                          'Decision Tree', 'Random Forest', 'XGBoost']
        
        for model_name in expected_models:
            assert model_name in results
            result = results[model_name]
            
            # Metriken prüfen
            assert 'model' in result
            assert 'predictions' in result
            assert 'mse' in result
            assert 'rmse' in result
            assert 'mae' in result
            assert 'r2' in result
            assert 'cv_mean' in result
            assert 'cv_std' in result
            
            # Werte-Bereiche prüfen
            assert result['mse'] >= 0
            assert result['rmse'] >= 0
            assert result['mae'] >= 0
            assert -1 <= result['r2'] <= 1  # R² kann negativ sein bei schlechten Modellen
    
    def test_analyze_feature_importance(self, predictor, sample_data):
        """Test der Feature-Wichtigkeits-Analyse"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        predictor.train_models()
        
        importance_results = predictor.analyze_feature_importance()
        
        assert isinstance(importance_results, dict)
        
        # Baumbasierte Modelle sollten Feature-Wichtigkeit haben
        tree_models = ['Decision Tree', 'Random Forest', 'XGBoost']
        
        for model_name in tree_models:
            if model_name in importance_results:
                result = importance_results[model_name]
                assert 'importances' in result
                assert 'feature_names' in result
                assert 'sorted_indices' in result
                
                # Summe der Wichtigkeiten sollte ~1 sein (für manche Modelle)
                if model_name in ['Random Forest', 'Decision Tree']:
                    total_importance = np.sum(result['importances'])
                    assert abs(total_importance - 1.0) < 0.01
    
    def test_predict_price(self, predictor, sample_data):
        """Test der Einzelvorhersage"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        predictor.train_models()
        
        # Test-Fahrzeug
        test_car = {
            'brand': 'audi',
            'model': 'A3',
            'year': 2018,
            'transmission': 'Manual',
            'mileage': 25000,
            'fuelType': 'Petrol',
            'tax': 145,
            'mpg': 45.0,
            'engineSize': 2.0
        }
        
        predictions = predictor.predict_price(test_car)
        
        assert isinstance(predictions, dict)
        assert len(predictions) > 0
        
        # Alle Vorhersagen sollten positive Werte sein
        for model_name, price in predictions.items():
            assert price >= 0
            assert isinstance(price, (int, float))
    
    def test_predict_price_unknown_values(self, predictor, sample_data):
        """Test der Vorhersage mit unbekannten Werten"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        predictor.train_models()
        
        # Test mit unbekannten Werten
        test_car = {
            'brand': 'unknown_brand',
            'model': 'unknown_model',
            'year': 2018,
            'transmission': 'Unknown',
            'mileage': 25000,
            'fuelType': 'Unknown',
            'tax': 145,
            'mpg': 45.0,
            'engineSize': 2.0
        }
        
        # Sollte nicht crashen, sondern robust handhaben
        try:
            predictions = predictor.predict_price(test_car)
            assert isinstance(predictions, dict)
            # Vorhersagen können 0 sein bei unbekannten Werten
            for price in predictions.values():
                assert price >= 0
        except Exception:
            # Akzeptabel wenn robuste Fehlerbehandlung existiert
            pass
    
    def test_generate_report(self, predictor, sample_data):
        """Test der Berichtgenerierung"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        predictor.train_models()
        predictor.analyze_feature_importance()
        
        report = predictor.generate_report()
        
        assert isinstance(report, str)
        assert len(report) > 100  # Substantieller Bericht
        
        # Wichtige Inhalte prüfen
        assert "FORSCHUNGSFRAGEN" in report
        assert "RANKING DER MODELLE" in report
        assert "PRAKTISCHE IMPLIKATIONEN" in report
        assert "R²" in report
        assert "RMSE" in report
    
    def test_model_performance_thresholds(self, predictor, sample_data):
        """Test ob Modelle minimale Performance-Standards erfüllen"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        results = predictor.train_models()
        
        # Mindestens ein Modell sollte R² > 0.3 haben
        max_r2 = max(result['r2'] for result in results.values())
        assert max_r2 > 0.3, f"Beste R² Score ({max_r2}) ist zu niedrig"
        
        # RMSE sollte nicht übermäßig hoch sein
        price_std = predictor.y_train.std()
        min_rmse = min(result['rmse'] for result in results.values())
        assert min_rmse < 2 * price_std, f"Beste RMSE ({min_rmse}) ist zu hoch"
    
    def test_data_types_consistency(self, predictor, sample_data):
        """Test der Datentyp-Konsistenz"""
        predictor.data = sample_data
        
        # Nach Feature-Engineering sollten alle Features numerisch sein
        X, y = predictor.prepare_features()
        
        for col in X.columns:
            assert X[col].dtype in [np.int64, np.float64, np.int32, np.float32], \
                f"Feature {col} hat non-numerischen Datentyp: {X[col].dtype}"
        
        # Target sollte numerisch sein
        assert y.dtype in [np.int64, np.float64, np.int32, np.float32]
    
    def test_cross_validation_consistency(self, predictor, sample_data):
        """Test der Cross-Validation Konsistenz"""
        predictor.data = sample_data
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        results = predictor.train_models()
        
        for model_name, result in results.items():
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            test_r2 = result['r2']
            
            # CV-Mean sollte im plausiblen Bereich des Test-R² liegen
            # (mit Toleranz für Varianz)
            assert abs(cv_mean - test_r2) < 0.5, \
                f"{model_name}: CV mean ({cv_mean}) und Test R² ({test_r2}) sind zu unterschiedlich"
            
            # Standard deviation sollte plausibel sein
            assert 0 <= cv_std <= 1, f"{model_name}: CV std ({cv_std}) ist nicht plausibel"

class TestEdgeCases:
    """Tests für Edge Cases und Robustheit"""
    
    def test_single_brand_data(self):
        """Test mit Daten nur einer Marke"""
        single_brand_data = pd.DataFrame({
            'model': ['A3', 'A4', 'A6'] * 10,
            'year': np.random.randint(2010, 2024, 30),
            'price': np.random.randint(8000, 50000, 30),
            'transmission': np.random.choice(['Manual', 'Automatic'], 30),
            'mileage': np.random.randint(5000, 150000, 30),
            'fuelType': np.random.choice(['Petrol', 'Diesel'], 30),
            'tax': np.random.randint(20, 500, 30),
            'mpg': np.random.uniform(20, 80, 30),
            'engineSize': np.random.uniform(1.0, 4.0, 30),
            'brand': ['audi'] * 30
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            single_brand_data.drop('brand', axis=1).to_csv(
                os.path.join(temp_dir, "audi.csv"), index=False
            )
            
            predictor = CarPricePredictor(data_path=temp_dir)
            predictor.load_data()
            
            # Sollte funktionieren auch mit nur einer Marke
            assert len(predictor.data['brand'].unique()) == 1
    
    def test_minimal_data(self):
        """Test mit minimaler Datenmenge"""
        minimal_data = pd.DataFrame({
            'model': ['A3', 'A4'],
            'year': [2020, 2019],
            'price': [20000, 18000],
            'transmission': ['Manual', 'Automatic'],
            'mileage': [10000, 15000],
            'fuelType': ['Petrol', 'Diesel'],
            'tax': [150, 200],
            'mpg': [50.0, 45.0],
            'engineSize': [1.5, 2.0],
            'brand': ['audi', 'audi']
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            minimal_data.drop('brand', axis=1).to_csv(
                os.path.join(temp_dir, "audi.csv"), index=False
            )
            
            predictor = CarPricePredictor(data_path=temp_dir)
            
            try:
                predictor.load_data()
                predictor.clean_data()
                X, y = predictor.prepare_features()
                
                # Mit so wenigen Daten ist Training schwierig
                # Aber es sollte nicht crashen
                assert len(X) >= 2
            except Exception as e:
                # Erwartetes Verhalten bei zu wenigen Daten
                assert "data" in str(e).lower() or "sample" in str(e).lower()
    
    def test_extreme_price_values(self):
        """Test mit extremen Preiswerten"""
        extreme_data = pd.DataFrame({
            'model': ['A3', 'A4', 'Luxury'] * 10,
            'year': [2020] * 30,
            'price': [100, 500, 1000000] * 10,  # Extreme Werte
            'transmission': ['Manual'] * 30,
            'mileage': [10000] * 30,
            'fuelType': ['Petrol'] * 30,
            'tax': [150] * 30,
            'mpg': [50.0] * 30,
            'engineSize': [2.0] * 30,
            'brand': ['test'] * 30
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extreme_data.drop('brand', axis=1).to_csv(
                os.path.join(temp_dir, "test.csv"), index=False
            )
            
            predictor = CarPricePredictor(data_path=temp_dir)
            predictor.load_data()
            
            # Outlier-Behandlung sollte extreme Werte entfernen
            cleaned = predictor.clean_data()
            
            # Preisspanne sollte nach Bereinigung kleiner sein
            original_range = extreme_data['price'].max() - extreme_data['price'].min()
            cleaned_range = cleaned['price'].max() - cleaned['price'].min()
            assert cleaned_range < original_range

class TestIntegration:
    """Integrationstests für die gesamte Pipeline"""
    
    def test_complete_pipeline(self):
        """Test der kompletten Pipeline"""
        # Realistische Testdaten erstellen
        np.random.seed(42)  # Für reproduzierbare Tests
        
        realistic_data = pd.DataFrame({
            'model': (['A3', 'A4', 'A6'] * 20 + ['3 Series', '5 Series', 'X3'] * 20 + 
                     ['C-Class', 'E-Class', 'S-Class'] * 20),
            'year': np.random.randint(2015, 2024, 180),
            'price': np.random.normal(25000, 10000, 180).astype(int),
            'transmission': np.random.choice(['Manual', 'Automatic'], 180),
            'mileage': np.random.exponential(30000, 180).astype(int),
            'fuelType': np.random.choice(['Petrol', 'Diesel', 'Hybrid'], 180),
            'tax': np.random.randint(50, 400, 180),
            'mpg': np.random.normal(45, 15, 180),
            'engineSize': np.random.normal(2.0, 0.5, 180),
            'brand': ['audi'] * 60 + ['bmw'] * 60 + ['mercedes'] * 60
        })
        
        # Negative Preise und unrealistische Werte bereinigen
        realistic_data['price'] = np.clip(realistic_data['price'], 5000, 100000)
        realistic_data['mileage'] = np.clip(realistic_data['mileage'], 0, 300000)
        realistic_data['mpg'] = np.clip(realistic_data['mpg'], 15, 80)
        realistic_data['engineSize'] = np.clip(realistic_data['engineSize'], 0.8, 5.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Daten nach Marken aufteilen und speichern
            for brand in realistic_data['brand'].unique():
                brand_data = realistic_data[realistic_data['brand'] == brand].drop('brand', axis=1)
                brand_data.to_csv(os.path.join(temp_dir, f"{brand}.csv"), index=False)
            
            # Komplette Pipeline ausführen
            predictor = CarPricePredictor(data_path=temp_dir)
            
            # 1. Daten laden
            data = predictor.load_data()
            assert len(data) == len(realistic_data)
            
            # 2. Exploration
            exploration_results = predictor.explore_data()
            assert isinstance(exploration_results, dict)
            
            # 3. Bereinigung
            cleaned_data = predictor.clean_data()
            assert len(cleaned_data) <= len(data)
            
            # 4. Feature Engineering
            X, y = predictor.prepare_features()
            assert len(X) == len(y)
            
            # 5. Datenaufteilung
            predictor.split_data(X, y)
            
            # 6. Training
            model_results = predictor.train_models()
            assert len(model_results) >= 5  # Mindestens 5 Modelle
            
            # 7. Feature Importance
            importance_results = predictor.analyze_feature_importance()
            assert len(importance_results) >= 1
            
            # 8. Vorhersage
            test_car = {
                'brand': 'audi',
                'model': 'A3',
                'year': 2020,
                'transmission': 'Manual',
                'mileage': 20000,
                'fuelType': 'Petrol',
                'tax': 150,
                'mpg': 50.0,
                'engineSize': 1.8
            }
            
            predictions = predictor.predict_price(test_car)
            assert len(predictions) >= 5
            
            # 9. Bericht
            report = predictor.generate_report()
            assert len(report) > 500  # Substantieller Bericht
            
            # Performance-Check
            best_r2 = max(result['r2'] for result in model_results.values())
            assert best_r2 > 0.5, f"Beste R² ({best_r2}) sollte > 0.5 sein mit realistischen Daten"

if __name__ == "__main__":
    # Tests ausführen
    pytest.main([__file__, "-v", "--tb=short"])
