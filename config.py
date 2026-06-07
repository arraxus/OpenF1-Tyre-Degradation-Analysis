from pathlib import Path

# ŚCIEŻKI I KATALOGI
RAW  = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# KONFIGURACJA API (data_download.py)
BASE = "https://api.openf1.org/v1"
# Rate limiting
API_RATE_LIMIT_PER_SECOND = 3
API_RATE_LIMIT_PER_MINUTE = 30

# KONFIGURACJA OKIEN (windows.py)
WINDOW_SEC    = 10        # długość okna [s]
FUEL_START_KG = 100.0     # masa startowa paliwa [kg]
FUEL_LOSS_S   = 0.030     # korekta: 0.03 s / kg paliwa
SC_BUFFER_LAPS   = 2      # okrążenia bufor wokół SC/VSC
PIT_BUFFER_LAPS  = 1      # okrążenia bufor wokół pit-stopu

# CECHY I ETYKIETY (features.py)
# Progi stanów opony (etykiety)
CLIFF_THRESHOLD_PCT  = 1.5   # % spadku znorm. czasu → Stan 2
RISK_THRESHOLD_PCT   = 0.5   # % spadku znorm. czasu → Stan 1
# Parametry okien rolling
BRAKE_ENERGY_WINDOW  = "60s" # okno rolling dla proxy temperatury
SPEED_VAR_WINDOW     = "30s" # okno rolling dla wariancji prędkości
LAG_STEPS            = 3     # ile kroków wstecz
# Kolumny metadanych / kolumny do wykluczenia przy tworzeniu macierzy cech
METADATA_EXCLUDE = {
    "ts", "date", "session_key", "driver_number", "stint_number",
    "lap_number", "compound", "target", "lap_time_corr",
    "fuel_mass_kg", "stint_median_lap", "n_samples",
    "location", "gp_name", "driver"
}

# MODELE (models.py)
# TimeSeriesSplit
N_SPLITS      = 3      # liczba foldów
RANDOM_STATE  = 42     # seed dla reproducibility
# Optuna
OPTUNA_TRIALS = 50     # liczba prób optymalizacji hiperparametrów

# SELEKCJA WYŚCIGÓW (pipeline_full.py)
# Mapowanie lokacji na nazwy Grand Prix
TRACK_MAP = {
    # GP Bahrajnu
    "Sakhir":            "Bahrain GP",
    # GP Emilii-Romanii
    "Imola":             "Emilia Romagna GP",
    # GP Monako
    "Monaco":            "Monaco GP",
    # GP Belgii
    "Spa-Francorchamps": "Belgian GP",
    # GP Holandii
    "Zandvoort":         "Dutch GP",
    # GP Włoch (Monza)
    "Monza":             "Italian GP",
    # GP Japonii
    "Suzuka":            "Japanese GP",
    # GP Austrii
    "Spielberg":         "Austrian GP",
    # GP W. Brytanii
    "Silverstone":       "British GP",
    # GP Abu Zabi
    "Yas Island":        "Abu Dhabi GP",
}
TARGET_GPS = set(TRACK_MAP.values())  # 10 unikalnych GP