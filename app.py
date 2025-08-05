"""
Streamlit Web-App für die Gebrauchtwagen-Preisvorhersage

Diese Anwendung bietet eine benutzerfreundliche Oberfläche zur Vorhersage
von Gebrauchtwagenpreisen basierend auf verschiedenen ML-Modellen.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from car_price_predictor import CarPricePredictor

# Streamlit Konfiguration
st.set_page_config(
    page_title="Gebrauchtwagen-Preisvorhersage",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictor():
    """Lädt und trainiert den Predictor (mit Caching für Performance)"""
    predictor = CarPricePredictor()
    
    # Prüfen ob bereits trainierte Modelle existieren
    if os.path.exists('trained_models.pkl'):
        try:
            with open('trained_models.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                predictor.models = saved_data['models']
                predictor.model_results = saved_data['model_results'] 
                predictor.label_encoders = saved_data['label_encoders']
                predictor.scaler = saved_data['scaler']
                predictor.X_train = saved_data['X_train']
                predictor.X_test = saved_data.get('X_test')
                predictor.y_train = saved_data.get('y_train')
                predictor.y_test = saved_data.get('y_test')
                predictor.data = saved_data['data']
                predictor.feature_importance = saved_data.get('feature_importance', {})
                
                # Prüfen ob alle wichtigen Daten vorhanden sind
                if predictor.y_test is None or predictor.X_test is None:
                    st.warning("⚠️ Unvollständige Cached-Daten gefunden. Führe vollständiges Training durch...")
                    # Cache löschen und neu trainieren
                    os.remove('trained_models.pkl')
                    return load_predictor()
                
                return predictor
        except Exception as e:
            st.warning(f"⚠️ Fehler beim Laden der Cached-Daten: {e}. Führe neues Training durch...")
            if os.path.exists('trained_models.pkl'):
                os.remove('trained_models.pkl')
    
    # Ansonsten neu trainieren
    with st.spinner("Modelle werden trainiert... Dies kann einige Minuten dauern."):
        predictor.load_data()
        predictor.clean_data()
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        predictor.train_models()
        predictor.analyze_feature_importance()
        
        # Speichern für zukünftige Verwendung
        save_data = {
            'models': predictor.models,
            'model_results': predictor.model_results,
            'label_encoders': predictor.label_encoders,
            'scaler': predictor.scaler,
            'X_train': predictor.X_train,
            'X_test': predictor.X_test,
            'y_train': predictor.y_train,
            'y_test': predictor.y_test,
            'data': predictor.data,
            'feature_importance': getattr(predictor, 'feature_importance', {})
        }
        with open('trained_models.pkl', 'wb') as f:
            pickle.dump(save_data, f)
    
    return predictor

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Gebrauchtwagen-Preisvorhersage</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Wählen Sie eine Seite:",
        ["🏠 Startseite", "🔮 Preisvorhersage", "📊 Datenanalyse", "📈 Modell-Vergleich", "ℹ️ Über das Projekt"]
    )
    
    # Predictor laden
    predictor = load_predictor()
    
    if page == "🏠 Startseite":
        show_home_page(predictor)
    elif page == "🔮 Preisvorhersage":
        show_prediction_page(predictor)
    elif page == "📊 Datenanalyse":
        show_analysis_page(predictor)
    elif page == "📈 Modell-Vergleich":
        show_model_comparison_page(predictor)
    elif page == "ℹ️ Über das Projekt":
        show_about_page()

def show_home_page(predictor):
    """Zeigt die Startseite"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(
            "images/car.png",
            caption="Machine Learning für Gebrauchtwagenpreise"
        )


    
    st.markdown("## 🎯 Projektübersicht")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Forschungsziele
        
        1. **Algorithmus-Vergleich**: Welche ML-Methoden eignen sich am besten?
        2. **Feature-Analyse**: Welche Merkmale beeinflussen den Preis am stärksten?
        3. **Genauigkeits-Bewertung**: Wie präzise sind die Vorhersagen?
        4. **Markt-Implikationen**: Welche praktischen Erkenntnisse ergeben sich?
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Implementierte Modelle
        
        - **Linear Regression**: Baseline-Modell
        - **Ridge/Lasso**: Regularisierte lineare Modelle  
        - **Decision Tree**: Interpretierbare Baumstruktur
        - **Random Forest**: Ensemble-Methode
        - **XGBoost**: Gradient Boosting
        """)
    
    # Dataset Statistiken
    if predictor.data is not None:
        st.markdown("## 📈 Dataset-Übersicht")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Gesamtfahrzeuge", f"{len(predictor.data):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Durchschnittspreis", f"£{predictor.data['price'].mean():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Marken", predictor.data['brand'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Preisspanne", f"£{predictor.data['price'].min():,.0f} - £{predictor.data['price'].max():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_page(predictor):
    """Zeigt die Preisvorhersage-Seite"""
    
    st.markdown('<h2 class="sub-header">🔮 Preisvorhersage für Ihr Fahrzeug</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Fahrzeugdaten eingeben")
        
        # Eingabefelder
        brand_options = sorted(predictor.data['brand'].unique()) if predictor.data is not None else ['bmw', 'audi', 'mercedes']
        brand = st.selectbox("Marke", brand_options)
        
        # Modelle basierend auf Marke filtern
        if predictor.data is not None:
            brand_models = predictor.data[predictor.data['brand'] == brand]['model'].unique()
            model = st.selectbox("Modell", sorted(brand_models))
        else:
            model = st.text_input("Modell", value="3 Series")
        
        year = st.slider("Baujahr", min_value=2000, max_value=2024, value=2018)
        
        transmission = st.selectbox("Getriebe", ["Manual", "Automatic", "Semi-Auto"])
        
        mileage = st.number_input("Laufleistung (Meilen)", min_value=0, max_value=200000, value=25000, step=1000)
        
        fuel_type = st.selectbox("Kraftstoffart", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])
        
        tax = st.number_input("Steuer (£)", min_value=0, max_value=1000, value=145)
        
        mpg = st.number_input("Verbrauch (MPG)", min_value=10.0, max_value=100.0, value=45.0, step=0.1)
        
        engine_size = st.number_input("Hubraum (L)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    
    with col2:
        st.markdown("### Preisvorhersage")
        
        if st.button("💰 Preis vorhersagen", type="primary"):
            # Vorhersage erstellen
            car_features = {
                'brand': brand,
                'model': model,
                'year': year,
                'transmission': transmission,
                'mileage': mileage,
                'fuelType': fuel_type,
                'tax': tax,
                'mpg': mpg,
                'engineSize': engine_size
            }
            
            predictions = predictor.predict_price(car_features)
            
            # Beste Vorhersage hervorheben
            best_model = max(predictor.model_results.items(), key=lambda x: x[1]['r2'])
            best_prediction = predictions[best_model[0]]
            
            st.markdown(f'''
            <div class="prediction-box">
                <h3>🎯 Empfohlener Preis</h3>
                <h1>£{best_prediction:.0f}</h1>
                <p>Basierend auf {best_model[0]} (R² = {best_model[1]['r2']:.3f})</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Alle Vorhersagen anzeigen
            st.markdown("### 📊 Vorhersagen aller Modelle")
            
            pred_df = pd.DataFrame(list(predictions.items()), columns=['Modell', 'Vorhersage'])
            pred_df['Vorhersage'] = pred_df['Vorhersage'].round(0).astype(int)
            pred_df = pred_df.sort_values('Vorhersage', ascending=False)
            
            # Balkendiagramm
            fig = px.bar(pred_df, x='Modell', y='Vorhersage', 
                        title="Preisvorhersagen aller Modelle",
                        color='Vorhersage',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabelle
            st.dataframe(pred_df, use_container_width=True)

def show_analysis_page(predictor):
    """Zeigt die Datenanalyse-Seite"""
    
    st.markdown('<h2 class="sub-header">📊 Explorative Datenanalyse</h2>', unsafe_allow_html=True)
    
    if predictor.data is None:
        st.error("Keine Daten verfügbar")
        return
    
    # Dataset Übersicht
    st.markdown("### 📋 Dataset-Informationen")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(predictor.data.describe())
    
    with col2:
        missing_data = predictor.data.isnull().sum()
        if missing_data.sum() > 0:
            st.write("Fehlende Werte:")
            st.write(missing_data[missing_data > 0])
        else:
            st.success("✅ Keine fehlenden Werte!")
    
    # Visualisierungen
    st.markdown("### 📈 Datenverteilungen")
    
    # Preisverteilung
    fig1 = px.histogram(predictor.data, x='price', nbins=50, 
                       title="Preisverteilung")
    fig1.update_layout(xaxis_title="Preis (£)", yaxis_title="Anzahl")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Markenverteilung
    brand_counts = predictor.data['brand'].value_counts().head(10)
    fig2 = px.bar(x=brand_counts.index, y=brand_counts.values,
                  title="Top 10 Marken nach Anzahl")
    fig2.update_layout(xaxis_title="Marke", yaxis_title="Anzahl Fahrzeuge")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Korrelationsanalyse
    st.markdown("### 🔗 Korrelationsanalyse")
    
    numeric_cols = predictor.data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = predictor.data[numeric_cols].corr()
        
        fig3 = px.imshow(corr_matrix, 
                        title="Korrelationsmatrix numerischer Features",
                        color_continuous_scale='RdBu',
                        aspect='auto')
        st.plotly_chart(fig3, use_container_width=True)
    
    # Scatter Plots
    st.markdown("### 🎯 Feature vs. Preis Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'year' in predictor.data.columns:
            fig4 = px.scatter(predictor.data, x='year', y='price',
                             title="Baujahr vs. Preis", opacity=0.6)
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        if 'mileage' in predictor.data.columns:
            fig5 = px.scatter(predictor.data, x='mileage', y='price',
                             title="Laufleistung vs. Preis", opacity=0.6)
            st.plotly_chart(fig5, use_container_width=True)

def show_model_comparison_page(predictor):
    """Zeigt die Modell-Vergleichsseite"""
    
    st.markdown('<h2 class="sub-header">📈 Modell-Performance Vergleich</h2>', unsafe_allow_html=True)
    
    # Debug Information (nur in Entwicklung sichtbar)
    with st.expander("🔧 Debug Info (für Entwickler)", expanded=False):
        st.write("Verfügbare Predictor Attribute:")
        st.write(f"- models: {'✅' if hasattr(predictor, 'models') and predictor.models else '❌'}")
        st.write(f"- model_results: {'✅' if hasattr(predictor, 'model_results') and predictor.model_results else '❌'}")
        st.write(f"- y_test: {'✅' if hasattr(predictor, 'y_test') and predictor.y_test is not None else '❌'}")
        st.write(f"- X_test: {'✅' if hasattr(predictor, 'X_test') and predictor.X_test is not None else '❌'}")
        st.write(f"- feature_importance: {'✅' if hasattr(predictor, 'feature_importance') and predictor.feature_importance else '❌'}")
        
        if hasattr(predictor, 'y_test') and predictor.y_test is not None:
            st.write(f"- y_test Shape: {predictor.y_test.shape}")
    
    if not predictor.model_results:
        st.error("Keine Modell-Ergebnisse verfügbar")
        return
    
    # Performance Metriken Tabelle
    st.markdown("### 📊 Performance-Übersicht")
    
    results_data = []
    for name, results in predictor.model_results.items():
        results_data.append({
            'Modell': name,
            'R² Score': f"{results['r2']:.4f}",
            'RMSE (£)': f"{results['rmse']:.2f}",
            'MAE (£)': f"{results['mae']:.2f}",
            'CV R² Mean': f"{results['cv_mean']:.4f}",
            'CV R² Std': f"{results['cv_std']:.4f}"
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('R² Score', ascending=False)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualisierungen
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score Vergleich
        models = list(predictor.model_results.keys())
        r2_scores = [predictor.model_results[model]['r2'] for model in models]
        
        fig1 = px.bar(x=models, y=r2_scores,
                     title="R² Score Vergleich",
                     color=r2_scores,
                     color_continuous_scale='viridis')
        fig1.update_layout(xaxis_title="Modell", yaxis_title="R² Score")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # RMSE Vergleich
        rmse_scores = [predictor.model_results[model]['rmse'] for model in models]
        
        fig2 = px.bar(x=models, y=rmse_scores,
                     title="RMSE Vergleich (niedrigere Werte = besser)",
                     color=rmse_scores,
                     color_continuous_scale='Reds_r')
        fig2.update_layout(xaxis_title="Modell", yaxis_title="RMSE (£)")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Feature Importance
    if predictor.feature_importance and 'Random Forest' in predictor.feature_importance:
        st.markdown("### 🎯 Feature-Wichtigkeit (Random Forest)")
        
        rf_importance = predictor.feature_importance['Random Forest']
        importances = rf_importance['importances']
        feature_names = rf_importance['feature_names']
        indices = rf_importance['sorted_indices']
        
        top_n = min(15, len(indices))
        top_features = [feature_names[indices[i]] for i in range(top_n)]
        top_importances = [importances[indices[i]] for i in range(top_n)]
        
        fig3 = px.bar(x=top_importances, y=top_features, orientation='h',
                     title=f"Top {top_n} wichtigste Features",
                     color=top_importances,
                     color_continuous_scale='viridis')
        fig3.update_layout(xaxis_title="Wichtigkeit", yaxis_title="Feature")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Prediction vs Actual Plots
    st.markdown("### 🎯 Vorhersage vs. Tatsächliche Werte")
    
    # Prüfen ob Test-Daten verfügbar sind
    if hasattr(predictor, 'y_test') and predictor.y_test is not None:
        best_models = ['Random Forest', 'XGBoost']
        
        for model_name in best_models:
            if model_name in predictor.model_results and 'predictions' in predictor.model_results[model_name]:
                y_pred = predictor.model_results[model_name]['predictions']
                r2 = predictor.model_results[model_name]['r2']
                
                # Scatter plot
                fig = px.scatter(x=predictor.y_test, y=y_pred, 
                               title=f"{model_name}: Vorhersage vs. Tatsächliche Werte (R² = {r2:.3f})",
                               labels={'x': 'Tatsächliche Preise (£)', 'y': 'Vorhergesagte Preise (£)'},
                               opacity=0.6)
                
                # Ideale Linie hinzufügen
                min_val = min(predictor.y_test.min(), y_pred.min())
                max_val = max(predictor.y_test.max(), y_pred.max())
                fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                             line=dict(color="red", width=2, dash="dash"))
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Test-Daten nicht verfügbar. Vorhersage vs. Tatsächliche Werte können nicht angezeigt werden.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Modelle neu trainieren", type="primary"):
                # Cache löschen und Seite neu laden
                if os.path.exists('trained_models.pkl'):
                    os.remove('trained_models.pkl')
                st.rerun()

def show_about_page():
    """Zeigt die Über-Seite"""
    
    st.markdown('<h2 class="sub-header">ℹ️ Über das Projekt</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Projektbeschreibung
    
    Diese Anwendung wurde entwickelt, um die **Preisvorhersage von Gebrauchtwagen** mithilfe verschiedener 
    Machine Learning-Algorithmen zu untersuchen und zu bewerten.
    
    ### 🔬 Forschungsfragen
    
    1. **Algorithmus-Vergleich**: Welche ML-Algorithmen eignen sich am besten für die Preisvorhersage?
    2. **Feature-Analyse**: Welche Fahrzeugmerkmale haben den größten Einfluss auf den Preis?
    3. **Genauigkeits-Bewertung**: Wie genau können Preise mit den entwickelten Modellen vorhergesagt werden?
    4. **Markt-Implikationen**: Welche praktischen Implikationen ergeben sich für den Gebrauchtwagenmarkt?
    
    ### 🛠️ Technischer Stack
    
    - **Python 3.10+**: Programmiersprache
    - **Streamlit**: Web-Framework für die Benutzeroberfläche
    - **Scikit-learn**: Machine Learning Bibliothek
    - **XGBoost**: Gradient Boosting Framework
    - **Pandas/NumPy**: Datenverarbeitung
    - **Plotly**: Interaktive Visualisierungen
    
    ### 📊 Implementierte Modelle
    
    - **Linear Regression**: Baseline-Modell für lineare Beziehungen
    - **Ridge Regression**: Regularisierte lineare Regression
    - **Lasso Regression**: Regularisierte Regression mit Feature-Selektion
    - **Decision Tree**: Interpretierbare, baumbasierte Methode
    - **Random Forest**: Ensemble-Methode für robuste Vorhersagen
    - **XGBoost**: State-of-the-art Gradient Boosting
    
    ### 📈 Dataset
    
    Das Dataset umfasst Gebrauchtwagen verschiedener Marken mit folgenden Features:
    - **Marke & Modell**: Fahrzeugidentifikation
    - **Baujahr**: Alter des Fahrzeugs
    - **Laufleistung**: Gefahrene Kilometer/Meilen
    - **Getriebeart**: Manual/Automatik
    - **Kraftstoffart**: Benzin/Diesel/Hybrid/Elektro
    - **Hubraum**: Motorgröße
    - **Verbrauch**: Kraftstoffeffizienz
    - **Steuer**: Jährliche Kfz-Steuer
    
    ### 🚀 Anwendungsmöglichkeiten
    
    - **Händler**: Objektive Preisbewertung für den Einkauf
    - **Privatpersonen**: Faire Preisschätzung beim Kauf/Verkauf
    - **Versicherungen**: Risikobewertung und Schadenregulierung
    - **Finanzdienstleister**: Kreditbewertung für Fahrzeugfinanzierungen
    
    ### 📧 Kontakt
    
    Bei Fragen oder Anregungen wenden Sie sich bitte an das Entwicklungsteam.
    """)

if __name__ == "__main__":
    main()
