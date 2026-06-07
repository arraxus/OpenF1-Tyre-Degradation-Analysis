import requests
import polars as pl
import json
import time
from config import BASE, RAW, API_RATE_LIMIT_PER_SECOND, API_RATE_LIMIT_PER_MINUTE
from rate_limiter import RateLimiter

# Jedna globalna instancja — współdzielona przez cały pipeline
_limiter = RateLimiter(per_second=API_RATE_LIMIT_PER_SECOND, per_minute=API_RATE_LIMIT_PER_MINUTE)

def get(endpoint: str, params: dict, *, force: bool = False) -> list:
    """
    Pobiera dane z OpenF1, zapisuje cache JSON.
    force=True pomija cache i zawsze odpytuje API.
    """
    # Klucz cache z nazwy endpointu i parametrów
    cache_key = "_".join(
        [endpoint] + [f"{k}-{v}" for k, v in sorted(params.items())]
    )
    cache = RAW / f"{cache_key}.json"

    if cache.exists() and not force:
        return json.loads(cache.read_text())

    # Rate limiting przed każdym requestem do API
    _limiter.wait()

    try:
        r = requests.get(
            f"{BASE}/{endpoint}",
            params=params,
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data))
        return data

    except requests.HTTPError as e:
        # 429 = Too Many Requests — poczekaj i spróbuj raz jeszcze
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 60))
            print(f"  [429] Czekam {retry_after}s...")
            time.sleep(retry_after)
            return get(endpoint, params, force=True)
        raise

    except requests.RequestException as e:
        print(f"  [BŁĄD] {endpoint} {params}: {e}")
        return []

def get_race_sessions(year: int = 2025) -> pl.DataFrame:
    data = get("sessions", {"year": year, "session_name": "Race"})
    return pl.DataFrame(data).select([
        "session_key", "location", "country_name",
        "date_start", "date_end", "circuit_short_name"
    ]).sort("date_start")

def get_drivers(session_key: int) -> pl.DataFrame:
    data = get("drivers", {"session_key": session_key})
    return pl.DataFrame(data).select([
        "driver_number", "name_acronym", "team_name"
    ])

def get_car_data(session_key: int, driver_number: int) -> pl.DataFrame:
    data = get("car_data", {
        "session_key": session_key,
        "driver_number": driver_number
    })
    if not data:
        return pl.DataFrame()
    df = pl.DataFrame(data)
    # Konwersja timestamp i sortowanie
    df = df.with_columns(
        pl.col("date").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f+00:00", strict=False)
          .alias("ts")
    ).sort("ts")
    return df

def get_laps(session_key: int, driver_number: int) -> pl.DataFrame:
    data = get("laps", {
        "session_key": session_key,
        "driver_number": driver_number
    })
    if not data:
        return pl.DataFrame()
    return pl.DataFrame(data).sort("lap_number")

def get_stints(session_key: int, driver_number: int) -> pl.DataFrame:
    data = get("stints", {
        "session_key": session_key,
        "driver_number": driver_number
    })
    if not data:
        return pl.DataFrame()
    return pl.DataFrame(data).sort("stint_number")

def get_race_control(session_key: int) -> pl.DataFrame:
    """SC, VSC, yellow flags — do filtrowania okien.
    Budujemy DataFrame defensywnie: normalizujemy klucze i typy,
    aby uniknąć błędów przy niejednorodnym schemacie zwróconych JSON-ów.
    """
    data = get("race_control", {"session_key": session_key})
    if not data:
        return pl.DataFrame()

    # Normalizacja: zbierz wszystkie klucze występujące w rekordach
    keys = set()
    for rec in data:
        if isinstance(rec, dict):
            keys.update(rec.keys())

    # Funkcja pomocnicza do konwersji typów (dostosuj pola według API)
    def normalize_record(rec: dict) -> dict:
        nr = {}
        for k in keys:
            v = rec.get(k, None)
            # Konwertuj lap_number na int lub None
            if k == "lap_number":
                if v is None:
                    nr[k] = None
                else:
                    try:
                        nr[k] = int(v)
                    except Exception:
                        nr[k] = None
            # Możesz dodać tu inne konwersje pól (np. driver_number -> int)
            else:
                nr[k] = v
        return nr

    normalized = [normalize_record(rec if isinstance(rec, dict) else {}) for rec in data]

    # Spróbuj skonstruować DataFrame; jeśli błąd, wypisz diagnostykę
    try:
        # infer_schema_length zwiększone do rozmiaru danych - dokładna inferencja
        df = pl.DataFrame(normalized, infer_schema_length=len(normalized))
    except Exception as e:
        # diagnostyka: pokaz kilka pierwszych rekordów i typy wartości
        print(f"[WARN] Nie udało się zbudować DataFrame z race_control (session_key={session_key}): {e}")
        for i, rec in enumerate(normalized[:10]):
            print(f" REC[{i}]: {rec}")
            print({k: type(v).__name__ for k, v in rec.items()})
        # alternatywnie zbuduj DataFrame przez pandas (bardziej liberalne)
        try:
            import pandas as pd
            pdf = pd.DataFrame(normalized)
            df = pl.from_pandas(pdf)
            print("[INFO] Użyto konwersji przez pandas (mniej restrykcyjne dopasowanie typów).")
        except Exception as e2:
            print(f"[ERROR] Fallback przez pandas nie powiódł się: {e2}")
            return pl.DataFrame()

    # Parsuj timestamp jeśli istnieje kolumna z datą (nazywa się 'date' w JSON)
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f+00:00", strict=False)
            .alias("ts")
        )

    return df

def get_pit(session_key: int, driver_number: int) -> pl.DataFrame:
    data = get("pit", {
        "session_key": session_key,
        "driver_number": driver_number
    })
    if not data:
        return pl.DataFrame()
    return pl.DataFrame(data).sort("lap_number")

if __name__ == "__main__":
    sessions = get_race_sessions(2025)
    print(sessions)

    # Wybierz pierwsze GP do testów
    sk = sessions["session_key"][0]
    print(f"\nSession key: {sk}")

    drivers = get_drivers(sk)
    print(drivers)

    # Test dla jednego kierowcy
    dn = int(drivers["driver_number"][0])
    car = get_car_data(sk, dn)
    laps = get_laps(sk, dn)
    print(f"\ncar_data shape: {car.shape}")
    print(f"laps shape:     {laps.shape}")
    print(car.head(3))