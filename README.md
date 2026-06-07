# OpenF1 Tyre Degradation Analysis
Projekt analizuje zużycie opon i prognozuje ich przyszły stan oraz nagłe pogorszenia wydajności w warunkach wyścigowych Formuły 1. Zawiera proces pobierania danych, eksploracyjną analizę, ekstrakcję cech, trenowanie modeli oraz interpretację wyników (SHAP).

## Kolejność przeprowadzania analizy
1. Konfiguracja parametrów w `config.py` (ścieżki, ustawienia pipeline, treningu)
2. Spradzenie połączenia z API przez `data_download.py`
3. Pobranie i przetworzenie danych + inżynieria cech: `pipeline_full.py`
4. Analiza i interpretacja:
   - Eksploracyjna analiza: `eda.py`
   - Analiza cech: `features.py` (używane w pipeline)
   - Trening/modelowanie: `models.py`
   - SHAP/interpretacja: `shap_analysis.py`

## Struktura repozytorium
- `config.py` - ustawienia i parametry pipeline
- `data_download.py` - skrypty do pobierania i przygotowania surowych danych
- `eda.py` - skrypty eksploracyjnej analizy danych i wizualizacji
- `features.py` - generowanie cech / przetwarzanie szeregów czasowych
- `models.py` - trenowanie i ocena modeli predykcyjnych
- `pipeline_full.py` - skrypt orkiestrujący cały pipeline
- `shap_analysis.py` - analiza interpretowalności modeli (SHAP)
- `windows.py`, `rate_limiter.py` - pomocnicze moduły (okna czasowe, ograniczanie zapytań)

## Wymagania
- Python 3.13+
- Biblioteki w `requirements.txt`

## Wyniki
Model, wykresy i raporty są zapisywane lokalnie zgodnie z ustawieniami w `config.py`.

