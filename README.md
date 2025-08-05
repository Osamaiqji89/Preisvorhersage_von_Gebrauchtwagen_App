# Gebrauchtwagen-Preisvorhersage mit Machine Learning

🚗 **Intelligente Preisvorhersage für Gebrauchtwagen basierend auf Machine Learning**

## 🎯 Projektübersicht

Dieses Projekt implementiert verschiedene Machine Learning-Algorithmen zur Vorhersage von Gebrauchtwagenpreisen und beantwortet folgende Forschungsfragen:

1. **Welche ML-Algorithmen eignen sich am besten für die Preisvorhersage?**
2. **Welche Fahrzeugmerkmale haben den größten Einfluss auf den Preis?**
3. **Wie genau können Preise mit den entwickelten Modellen vorhergesagt werden?**
4. **Welche praktischen Implikationen ergeben sich für den Gebrauchtwagenmarkt?**

## 📊 Features

- **6 verschiedene ML-Modelle**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost
- **Umfassende Datenanalyse**: Explorative Datenanalyse mit statistischen Kennzahlen
- **Feature Engineering**: Automatische Erstellung relevanter Features (Alter, Laufleistung pro Jahr, etc.)
- **Interactive Web-App**: Streamlit-basierte Benutzeroberfläche
- **Umfassende Tests**: Pytest-basierte Testsuite mit >90% Abdeckung
- **Visualisierungen**: Interaktive Plotly-Diagramme und Matplotlib-Grafiken

## 🚀 Installation & Setup

### Voraussetzungen
- Python 3.10 oder höher
- Git

### 1. Repository klonen
```bash
git clone <repository-url>
cd Preisvorhersage_von_Gebrauchtwagen_App
```

### 2. Virtuelle Umgebung erstellen (empfohlen)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# oder
source .venv/bin/activate  # Linux/Mac
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Daten vorbereiten
Stellen Sie sicher, dass Ihre CSV-Dateien im `Daten/` Ordner liegen mit folgender Struktur:
```
Daten/
├── audi.csv
├── bmw.csv
├── mercedes.csv
└── ...weitere Marken-CSV-Dateien
```

## 💻 Verwendung

### Command Line Interface
```bash
# Vollständige Analyse ausführen
python car_price_predictor.py

# Tests ausführen
pytest test_car_price_predictor.py -v

# Web-App starten
streamlit run streamlit_app.py
```

### Programmierung API
```python
from car_price_predictor import CarPricePredictor

# Predictor initialisieren
predictor = CarPricePredictor(data_path="Daten")

# Komplette Analyse durchführen
predictor.run_complete_analysis()

# Einzelvorhersage
car_features = {
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

predictions = predictor.predict_price(car_features)
print(f"Vorhergesagter Preis: £{predictions['Random Forest']:.0f}")
```

## 📈 Datenformat

Erwartete CSV-Struktur für jede Automarke:
```csv
model,year,price,transmission,mileage,fuelType,tax,mpg,engineSize
A3,2018,15000,Manual,25000,Petrol,150,55.4,1.4
A4,2019,22000,Automatic,18000,Diesel,145,62.8,2.0
...
```

### Spalten-Beschreibung:
- **model**: Fahrzeugmodell (z.B. "A3", "3 Series")
- **year**: Baujahr (z.B. 2018)
- **price**: Preis in Pfund (z.B. 15000)
- **transmission**: Getriebeart ("Manual", "Automatic", "Semi-Auto")
- **mileage**: Laufleistung in Meilen (z.B. 25000)
- **fuelType**: Kraftstoffart ("Petrol", "Diesel", "Hybrid", "Electric")
- **tax**: Jährliche Kfz-Steuer in Pfund (z.B. 150)
- **mpg**: Kraftstoffverbrauch in Meilen pro Gallone (z.B. 55.4)
- **engineSize**: Hubraum in Litern (z.B. 1.4)

## 🧪 Tests

Das Projekt enthält eine umfassende Testsuite:

```bash
# Alle Tests ausführen
pytest

# Tests mit Coverage-Report
pytest --cov=car_price_predictor --cov-report=html

# Spezifische Tests
pytest test_car_price_predictor.py::TestCarPricePredictor::test_train_models -v
```

### Test-Kategorien:
- **Unit Tests**: Einzelne Funktionen und Methoden
- **Integration Tests**: Komplette Pipeline-Tests
- **Edge Case Tests**: Grenzfälle und Robustheit
- **Data Validation Tests**: Datenqualität und -konsistenz

## 📊 Ergebnisse & Performance

### Typische Modell-Performance:
- **Random Forest**: R² ≈ 0.85-0.92, RMSE ≈ £2,500-3,500
- **XGBoost**: R² ≈ 0.84-0.91, RMSE ≈ £2,600-3,600  
- **Linear Regression**: R² ≈ 0.75-0.85, RMSE ≈ £3,000-4,000

### Wichtigste Features (typisch):
1. **Fahrzeugalter** (25-35% Wichtigkeit)
2. **Laufleistung** (15-25% Wichtigkeit)
3. **Hubraum** (10-20% Wichtigkeit)
4. **Marke/Modell** (15-25% Wichtigkeit)
5. **Kraftstoffart** (5-15% Wichtigkeit)

## 🌐 Web-Application

Die Streamlit-App bietet folgende Features:

### 🏠 Startseite
- Projektübersicht und Forschungsfragen
- Dataset-Statistiken
- Navigation zu verschiedenen Bereichen

### 🔮 Preisvorhersage
- Interaktive Eingabemaske für Fahrzeugdaten
- Sofortige Preisvorhersage mit allen Modellen
- Konfidenzintervalle und Empfehlungen

### 📊 Datenanalyse
- Explorative Datenanalyse mit interaktiven Plots
- Korrelationsmatrizen und Verteilungsanalysen
- Feature-Statistiken

### 📈 Modell-Vergleich
- Performance-Metriken aller Modelle
- Feature-Wichtigkeit Visualisierungen
- Residual-Analysen

## 🛠️ Technischer Stack

- **Python 3.10+**: Hauptprogrammiersprache
- **Pandas/NumPy**: Datenverarbeitung und numerische Berechnungen
- **Scikit-learn**: Machine Learning Algorithmen
- **XGBoost**: Gradient Boosting Framework
- **Streamlit**: Web-App Framework
- **Plotly/Matplotlib**: Interaktive und statische Visualisierungen
- **Pytest**: Testing Framework

## 📁 Projektstruktur

```
Preisvorhersage_von_Gebrauchtwagen_App/
├── Daten/                          # CSV-Dateien mit Fahrzeugdaten
│   ├── audi.csv
│   ├── bmw.csv
│   └── ...
├── car_price_predictor.py          # Hauptklasse mit ML-Pipeline
├── streamlit_app.py                # Web-Application
├── utils.py                        # Utility-Funktionen
├── test_car_price_predictor.py     # Testsuite
├── requirements.txt                # Python-Dependencies
├── README.md                       # Diese Datei
└── Ausgabe/                        # Generierte Dateien
    ├── model_comparison.png        # Modell-Vergleichsdiagramme
    ├── feature_importance.png      # Feature-Wichtigkeit
    ├── predictions_vs_actual.png   # Vorhersage-Qualität
    ├── data_distributions.png      # Datenverteilungen
    └── analysebericht.txt           # Detaillierter Bericht
```

## 🔧 Konfiguration & Anpassungen

### Neue Modelle hinzufügen:
```python
# In car_price_predictor.py, Methode train_models()
from sklearn.ensemble import GradientBoostingRegressor

self.models['Gradient Boosting'] = GradientBoostingRegressor(
    n_estimators=100, 
    random_state=42
)
```

### Feature Engineering erweitern:
```python
# In prepare_features() Methode
# Neue Features basierend auf Domain-Wissen
X['luxury_brand'] = X['brand'].isin(['mercedes', 'bmw', 'audi']).astype(int)
X['fuel_efficiency_category'] = pd.cut(X['mpg'], bins=3, labels=['Low', 'Medium', 'High'])
```

## 📈 Performance-Optimierung

### Für große Datasets:
```python
# Sampling für schnellere Entwicklung
predictor = CarPricePredictor()
predictor.load_data()
sample_data = predictor.data.sample(n=10000, random_state=42)
predictor.data = sample_data
```

### Hyperparameter-Tuning:
```python
from sklearn.model_selection import GridSearchCV

# Beispiel für Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
```

## 🤝 Beitragen

1. Repository forken
2. Feature-Branch erstellen (`git checkout -b feature/neue-funktion`)
3. Änderungen committen (`git commit -am 'Neue Funktion hinzugefügt'`)
4. Branch pushen (`git push origin feature/neue-funktion`)
5. Pull Request erstellen

## 📝 Lizenz

Dieses Projekt ist für Bildungszwecke entwickelt worden.

## 📧 Kontakt

Bei Fragen oder Problemen:
- Issue auf GitHub erstellen
- E-Mail an das Entwicklungsteam

## 🙏 Danksagungen

- Kaggle für ähnliche Datasets als Inspiration
- Scikit-learn Community für exzellente Dokumentation
- Streamlit Team für das großartige Web-Framework

---

**Made with ❤️ for Data Science Education**
