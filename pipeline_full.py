import polars as pl
from data_download import get_race_sessions, get_drivers, get_race_control, get_quali_fastest_laps
from windows import process_driver
from features import build_feature_matrix, get_feature_columns
from config import PROC, TRACK_MAP, TARGET_GPS

def run_pipeline(year: int = 2025):
    sessions = get_race_sessions(year)

    # Mapuj location - nazwa GP, odfiltruj tylko wybrane
    sessions = sessions.with_columns(
        pl.col("location")
        .map_elements(lambda loc: TRACK_MAP.get(loc, None), return_dtype=pl.Utf8)
        .alias("gp_name")
    ).filter(pl.col("gp_name").is_not_null())

    # race_date: data wyścigu jako string ISO (YYYY-MM-DD)
    # używana jako klucz grup w chronological_split
    sessions = sessions.with_columns(
        pl.col("date_start").str.slice(0, 10).alias("race_date")
    )

    print(f"Znalezione GP ({sessions.shape[0]}/10):")
    print(sessions.select(["session_key", "location", "gp_name",
                           "date_start", "race_date"]))

    # Ostrzeżenie jeśli brakuje któregoś GP
    found_gps = set(sessions["gp_name"].to_list())
    missing = TARGET_GPS - found_gps
    if missing:
        print(f"\n⚠️  BRAKUJE GP: {missing}")

    all_windows = []

    for row in sessions.iter_rows(named=True):
        sk = row["session_key"]
        loc = row["location"]
        gp_name = row["gp_name"]
        race_date = row["race_date"]

        print(f"\n{'─'*50}")
        print(f"{gp_name} | {loc} (session_key={sk})")

        # Kwalifikacje — baseline toru
        print(f"  Pobieranie czasów kwalifikacji ({loc})...")
        quali_laps = get_quali_fastest_laps(loc, year)
        if quali_laps.shape[0] == 0:
            print("  [WARN] Brak danych kwalifikacji — normalizacja tylko wg mediany stintu")
        else:
            print(f"  Kwalifikacje: {quali_laps.shape[0]} kierowców, "
                  f"najszybszy: {quali_laps['quali_best_lap_s'].min():.3f}s")

        rc = get_race_control(sk)
        drivers = get_drivers(sk)

        for drow in drivers.iter_rows(named=True):
            dn   = drow["driver_number"]
            name = drow["name_acronym"]
            print(f"  Przetwarzam: {name} (#{dn})", end=" ")

            try:
                w = process_driver(sk, dn, rc)
                if w.shape[0] == 0:
                    print("→ brak danych")
                    continue

                w = w.with_columns([
                    pl.lit(loc).alias("location"),
                    pl.lit(gp_name).alias("gp_name"),
                    pl.lit(name).alias("driver"),
                    pl.lit(race_date).alias("race_date"),
                ])

                # Dołącz czas kwalifikacji tego kierowcy (jeśli dostępny)
                if quali_laps.shape[0] > 0:
                    w = w.join(
                        quali_laps.select(["driver_number", "quali_best_lap_s"]),
                        on="driver_number",
                        how="left",
                    )
                else:
                    w = w.with_columns(
                        pl.lit(None).cast(pl.Float64).alias("quali_best_lap_s")
                    )

                all_windows.append(w)
                print(f"→ {w.shape[0]} okien")

            except Exception as e:
                print(f"→ BŁĄD: {e}")

    if not all_windows:
        print("\nBrak danych!")
        return None

    # Łączenie
    raw = pl.concat(all_windows, how="diagonal")
    print(f"\nŁącznie okien surowych: {raw.shape[0]}")
    print(f"Okna z czasem kwalifikacji: "
          f"{raw['quali_best_lap_s'].drop_nulls().shape[0]} / {raw.shape[0]}")

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