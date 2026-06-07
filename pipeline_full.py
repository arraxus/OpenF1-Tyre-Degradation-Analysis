import polars as pl
from data_download import get_race_sessions, get_drivers, get_race_control
from windows import process_driver
from features import build_feature_matrix, get_feature_columns
from config import PROC, TRACK_MAP, TARGET_GPS

def run_pipeline(year: int = 2025):
    sessions = get_race_sessions(year)

    # Mapuj location → nazwa GP, odfiltruj tylko wybrane
    sessions = sessions.with_columns(
        pl.col("location")
        .map_elements(lambda loc: TRACK_MAP.get(loc, None), return_dtype=pl.Utf8)
        .alias("gp_name")
    ).filter(pl.col("gp_name").is_not_null())

    print(f"Znalezione GP ({sessions.shape[0]}/10):")
    print(sessions.select(["session_key", "location", "gp_name", "date_start"]))

    # Ostrzeżenie jeśli brakuje któregoś GP
    found_gps = set(sessions["gp_name"].to_list())
    missing = TARGET_GPS - found_gps
    if missing:
        print(f"\n⚠️  BRAKUJE GP: {missing}")
        print("Sprawdź powyżej jakie 'location' zwróciło API i uzupełnij TRACK_MAP")

    all_windows = []

    for row in sessions.iter_rows(named=True):
        sk  = row["session_key"]
        loc = row["location"]
        gp_name = row["gp_name"]
        print(f"\n{'─'*50}")
        print(f"{gp_name} | {loc} (session_key={sk})")

        rc = get_race_control(sk)
        drivers = get_drivers(sk)

        for drow in drivers.iter_rows(named=True):
            dn   = drow["driver_number"]
            name = drow["name_acronym"]
            print(f"  Przetwarzam: {name} (#{dn})", end=" ")

            try:
                w = process_driver(sk, dn, rc)
                if w.shape[0] > 0:
                    w = w.with_columns([
                        pl.lit(loc).alias("location"),
                        pl.lit(gp_name).alias("gp_name"),
                        pl.lit(name).alias("driver"),
                    ])
                    all_windows.append(w)
                    print(f"→ {w.shape[0]} okien")
                else:
                    print("→ brak danych")
            except Exception as e:
                print(f"→ BŁĄD: {e}")

    if not all_windows:
        print("\nBrak danych!")
        return None

    # Łączenie
    raw = pl.concat(all_windows, how="diagonal")
    print(f"\nŁącznie okien surowych: {raw.shape[0]}")

    # Inżynieria cech
    features = build_feature_matrix(raw)
    print(f"Po inżynierii cech:    {features.shape[0]} okien")
    print(f"Liczba cech:           {len(get_feature_columns(features))}")

    # Rozkład klas
    print("\nRozkład klas (target):")
    print(features.group_by("target").agg(pl.len().alias("count"))
            .sort("target"))

    # Zapis
    out = PROC / "features_all.parquet"
    features.write_parquet(out)
    print(f"\nZapisano: {out}")

    return features


if __name__ == "__main__":
    df = run_pipeline(2025)