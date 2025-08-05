@echo off
echo 🚗 Gebrauchtwagen-Preisvorhersage - Starter 🚗
echo ================================================

echo.
echo Wählen Sie eine Option:
echo.
echo [1] Demo ausführen (schnell)
echo [2] Vollständige Analyse
echo [3] Web-App starten (Streamlit)
echo [4] Tests ausführen
echo [5] Beenden
echo.

set /p choice="Ihre Wahl (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starte Demo...
    .venv\Scripts\python.exe demo.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starte vollständige Analyse...
    .venv\Scripts\python.exe car_price_predictor.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Starte Web-App...
    echo Öffnen Sie http://localhost:8501 im Browser
    .venv\Scripts\streamlit.exe run app.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Führe Tests aus...
    .venv\Scripts\python.exe -m pytest test_car_price_predictor.py -v
    goto end
)

if "%choice%"=="5" (
    echo Auf Wiedersehen!
    goto end
)

echo Ungültige Auswahl!

:end
echo.
pause
