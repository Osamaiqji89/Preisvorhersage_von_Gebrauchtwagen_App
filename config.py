"""
Konfigurationsdatei für die Gebrauchtwagen-Preisvorhersage

Zentrale Konfiguration aller Parameter und Einstellungen
"""

import os
from pathlib import Path

# Basis-Pfade
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Daten"
OUTPUT_DIR = BASE_DIR / "output"

# Erstelle Output-Verzeichnis falls nicht vorhanden
OUTPUT_DIR.mkdir(exist_ok=True)

# Datenparameter
DATA_CONFIG = {
    'data_path': str(DATA_DIR),
    'output_path': str(OUTPUT_DIR),
    'encoding': 'utf-8',
    'required_columns': [
        'model', 'year', 'price', 'transmission', 
        'mileage', 'fuelType', 'tax', 'mpg', 'engineSize'
    ],
    'exclude_files': ['unclean'],  # Dateien die "unclean" enthalten werden ausgeschlossen
    'min_samples_per_brand': 50,   # Minimale Anzahl Samples pro Marke
}

# Modell-Konfiguration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1,  # Alle CPU-Kerne verwenden
    
    # Modell-spezifische Parameter
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False
    },
    
    'ridge_regression': {
        'alpha': 1.0,
        'max_iter': 1000
    },
    
    'lasso_regression': {
        'alpha': 1.0,
        'max_iter': 1000
    },
    
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
}

# Feature Engineering Parameter
FEATURE_CONFIG = {
    'create_age_feature': True,
    'create_mileage_per_year': True,
    'create_efficiency_ratio': True,
    'create_power_category': True,
    'create_tax_per_liter': True,
    'create_binary_features': True,
    
    # Outlier-Behandlung
    'outlier_method': 'iqr',  # 'iqr' oder 'zscore'
    'outlier_threshold': 1.5,  # Für IQR-Methode
    
    # Feature-Skalierung
    'scale_features': True,
    'scaler_type': 'standard',  # 'standard', 'minmax', 'robust'
}

# Validierungs-Parameter
VALIDATION_CONFIG = {
    'price_min': 500,      # Minimaler plausibler Preis
    'price_max': 200000,   # Maximaler plausibler Preis
    'year_min': 1990,      # Ältestes Jahr
    'year_max': 2024,      # Aktuelles Jahr
    'mileage_max': 500000, # Maximale Laufleistung
    'mpg_min': 5,          # Minimaler Verbrauch
    'mpg_max': 100,        # Maximaler Verbrauch (unrealistisch hoch)
    'engine_size_max': 10, # Maximaler Hubraum
}

# Visualisierungs-Parameter
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_plots': True,
    'plot_formats': ['png', 'pdf'],  # Speicher-Formate
    
    # Plotly-spezifisch
    'plotly_theme': 'plotly_white',
    'show_plots': True,
}

# Streamlit App Konfiguration
STREAMLIT_CONFIG = {
    'page_title': "Gebrauchtwagen-Preisvorhersage",
    'page_icon': "🚗",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    
    # Cache-Einstellungen
    'cache_ttl': 3600,  # 1 Stunde in Sekunden
    'cache_allow_output_mutation': True,
    
    # UI-Einstellungen
    'show_progress_bar': True,
    'show_balloons_on_success': True,
    'max_upload_size': 200,  # MB
}

# Logging-Konfiguration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(OUTPUT_DIR / 'car_prediction.log'),
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
}

# Performance-Schwellenwerte
PERFORMANCE_THRESHOLDS = {
    'min_r2_score': 0.3,        # Minimaler R² Score
    'max_rmse_ratio': 2.0,      # RMSE/Std-Dev Verhältnis
    'min_cv_consistency': 0.1,  # Minimale CV-Konsistenz
    'max_training_time': 300,   # Maximale Trainingszeit in Sekunden
}

# API-Konfiguration (für zukünftige Entwicklung)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 4,
    'timeout': 30,
}

# Export-Konfiguration
EXPORT_CONFIG = {
    'model_format': 'pickle',  # 'pickle', 'joblib', 'json'
    'report_format': 'txt',    # 'txt', 'html', 'pdf'
    'export_predictions': True,
    'export_feature_importance': True,
    'export_model_metrics': True,
}

# Entwicklungs-Konfiguration
DEV_CONFIG = {
    'debug_mode': False,
    'sample_data_size': None,  # None für alle Daten, sonst Anzahl Samples
    'quick_mode': False,       # Reduzierte Modell-Parameter für schnelle Tests
    'verbose': True,
    'warnings_as_errors': False,
}

# Zusammenfassung aller Konfigurationen
CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'features': FEATURE_CONFIG,
    'validation': VALIDATION_CONFIG,
    'plots': PLOT_CONFIG,
    'streamlit': STREAMLIT_CONFIG,
    'logging': LOGGING_CONFIG,
    'performance': PERFORMANCE_THRESHOLDS,
    'api': API_CONFIG,
    'export': EXPORT_CONFIG,
    'dev': DEV_CONFIG,
}

def get_config(section: str = None):
    """
    Gibt Konfiguration zurück
    
    Args:
        section: Spezifische Sektion oder None für alle
        
    Returns:
        Dictionary mit Konfiguration
    """
    if section is None:
        return CONFIG
    
    return CONFIG.get(section, {})

def update_config(section: str, key: str, value):
    """
    Aktualisiert Konfigurationswert
    
    Args:
        section: Konfigurationssektion
        key: Konfigurationsschlüssel
        value: Neuer Wert
    """
    if section in CONFIG:
        CONFIG[section][key] = value
    else:
        raise ValueError(f"Unbekannte Konfigurationssektion: {section}")

# Umgebungsvariablen überschreiben Standardkonfiguration
def load_env_config():
    """Lädt Konfiguration aus Umgebungsvariablen"""
    
    # Datenverzeichnis
    if 'CAR_DATA_PATH' in os.environ:
        CONFIG['data']['data_path'] = os.environ['CAR_DATA_PATH']
    
    # Debug-Modus
    if 'CAR_DEBUG' in os.environ:
        CONFIG['dev']['debug_mode'] = os.environ['CAR_DEBUG'].lower() == 'true'
    
    # Test-Größe
    if 'CAR_TEST_SIZE' in os.environ:
        CONFIG['model']['test_size'] = float(os.environ['CAR_TEST_SIZE'])
    
    # Anzahl CPU-Kerne
    if 'CAR_N_JOBS' in os.environ:
        CONFIG['model']['n_jobs'] = int(os.environ['CAR_N_JOBS'])

# Konfiguration beim Import laden
load_env_config()
