"""
Demo-Skript für die Gebrauchtwagen-Preisvorhersage

Einfaches Skript um die Funktionalität zu demonstrieren
"""

from car_price_predictor import CarPricePredictor
import pandas as pd

def demo():
    print("🚗 GEBRAUCHTWAGEN-PREISVORHERSAGE DEMO 🚗")
    print("=" * 60)
    
    # Predictor erstellen
    predictor = CarPricePredictor()
    
    print("📊 Lade Daten...")
    predictor.load_data()
    
    print(f"✅ {len(predictor.data):,} Fahrzeuge aus {predictor.data['brand'].nunique()} Marken geladen")
    print(f"💰 Durchschnittspreis: £{predictor.data['price'].mean():.0f}")
    
    print("\n🔧 Bereinige Daten und erstelle Features...")
    predictor.clean_data()
    X, y = predictor.prepare_features()
    predictor.split_data(X, y)
    
    print("\n🤖 Trainiere Modelle (nur die besten)...")
    
    # Erst die Modelle initialisieren
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    models_to_train = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=50, max_depth=6, verbosity=0)
    }
    
    # Training mit reduzierter Ausgabe
    results = {}
    for name, model in models_to_train.items():
        print(f"  Trainiere {name}...")
        
        model.fit(predictor.X_train, predictor.y_train)
        y_pred = model.predict(predictor.X_test)
        
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np
        
        r2 = r2_score(predictor.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(predictor.y_test, y_pred))
        
        results[name] = {'r2': r2, 'rmse': rmse, 'model': model}
        print(f"    R² = {r2:.3f}, RMSE = £{rmse:.0f}")
    
    predictor.models = models_to_train
    predictor.model_results = results
    
    print("\n🎯 DEMO-VORHERSAGEN:")
    print("-" * 40)
    
    # Verschiedene Demo-Fahrzeuge
    demo_cars = [
        {
            'name': 'BMW 3 Series (2020, neu)',
            'features': {
                'brand': 'bmw',
                'model': '3 Series', 
                'year': 2020,
                'transmission': 'Automatic',
                'mileage': 5000,
                'fuelType': 'Petrol',
                'tax': 145,
                'mpg': 45.0,
                'engineSize': 2.0
            }
        },
        {
            'name': 'Ford Focus (2016, gebraucht)',
            'features': {
                'brand': 'ford',
                'model': 'Focus',
                'year': 2016,
                'transmission': 'Manual',
                'mileage': 60000,
                'fuelType': 'Petrol',
                'tax': 125,
                'mpg': 50.0,
                'engineSize': 1.5
            }
        },
        {
            'name': 'Audi A4 (2019, Diesel)',
            'features': {
                'brand': 'audi',
                'model': 'A4',
                'year': 2019,
                'transmission': 'Automatic',
                'mileage': 20000,
                'fuelType': 'Diesel',
                'tax': 145,
                'mpg': 60.0,
                'engineSize': 2.0
            }
        }
    ]
    
    for car in demo_cars:
        predictions = predictor.predict_price(car['features'])
        print(f"\n🚙 {car['name']}:")
        for model_name, price in predictions.items():
            print(f"  {model_name}: £{price:,.0f}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO ABGESCHLOSSEN!")
    print("\nFür die vollständige Analyse führen Sie aus:")
    print("  python car_price_predictor.py")
    print("\nFür die Web-App führen Sie aus:")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    demo()
