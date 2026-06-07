import bisect
import polars as pl
from config import WINDOW_SEC, FUEL_START_KG, FUEL_LOSS_S, SC_BUFFER_LAPS, PIT_BUFFER_LAPS
from data_download import (
    get_car_data, get_laps, get_stints, get_pit
)

def build_fuel_correction(laps: pl.DataFrame) -> pl.DataFrame:
    # Dodaje kolumny: fuel_mass i lap_time_corr
    n_laps = laps.shape[0]
    avg_consumption = FUEL_START_KG / max(n_laps, 1)

    laps = laps.with_columns([
        (FUEL_START_KG - pl.col("lap_number") * avg_consumption)
          .clip(0, FUEL_START_KG)
          .alias("fuel_mass_kg"),
    ])
    laps = laps.with_columns([
        (pl.col("lap_duration") - pl.col("fuel_mass_kg") * FUEL_LOSS_S)
          .alias("lap_time_corr")
    ])
    return laps


def get_bad_laps(
    laps: pl.DataFrame,
    stints: pl.DataFrame,
    race_control: pl.DataFrame,
    pit: pl.DataFrame
) -> set:
    #Zwraca zbiór numerów okrążeń do odrzucenia
    bad = set()

    # In-lap i out-lap per stint
    if stints.shape[0] > 0 and "lap_start" in stints.columns and "lap_end" in stints.columns:
        for row in stints.iter_rows(named=True):
            bad.add(row["lap_start"])         # out-lap
            bad.add(row.get("lap_end", -1))   # in-lap

    # Okrążenia pit-stop
    if pit.shape[0] > 0 and "lap_number" in pit.columns:
        for lap in pit["lap_number"].to_list():
            for offset in range(-PIT_BUFFER_LAPS, PIT_BUFFER_LAPS + 1):
                bad.add(lap + offset)

    # Okrążenia SC/VSC
    sc_flags = {"SAFETY CAR", "VIRTUAL SAFETY CAR", "YELLOW"}
    if race_control.shape[0] > 0:
        rc_sc = race_control.filter(
            pl.col("flag").is_in(list(sc_flags)) |
            pl.col("category").str.contains("(?i)safety")
        )
        if rc_sc.shape[0] > 0 and "lap_number" in rc_sc.columns:
            for lap in rc_sc["lap_number"].drop_nulls().to_list():
                for offset in range(-SC_BUFFER_LAPS, SC_BUFFER_LAPS + 1):
                    bad.add(int(lap) + offset)

    bad.discard(None)
    return bad


def assign_lap_to_window(car: pl.DataFrame, laps: pl.DataFrame) -> pl.DataFrame:
    # Dodaje lap_number i stint_number do każdej próbki car_data
    if laps.shape[0] == 0:
        return car.with_columns([
            pl.lit(None).cast(pl.Int32).alias("lap_number"),
            pl.lit(None).cast(pl.Int32).alias("stint_number"),
        ])

    # Granice okrążeń z laps
    lap_bounds = laps.select([
        "lap_number",
        pl.col("date_start").str.to_datetime(
            format="%Y-%m-%dT%H:%M:%S%.f+00:00",
            strict=False
        ).alias("lap_ts_start"),
    ]).drop_nulls()

    # Sortuj car_data i lap_bounds
    car = car.sort("ts")
    lap_ts = lap_bounds.sort("lap_ts_start")

    # Join przez search_sorted (Polars >= 0.19)
    lap_starts = lap_ts["lap_ts_start"].to_list()
    lap_nums   = lap_ts["lap_number"].to_list()

    def find_lap(ts):
        idx = bisect.bisect_right(lap_starts, ts) - 1
        return lap_nums[idx] if idx >= 0 else None

    car = car.with_columns(
        pl.col("ts").map_elements(find_lap, return_dtype=pl.Int32).alias("lap_number")
    )
    return car


def make_windows(car: pl.DataFrame) -> pl.DataFrame:
    # Agreguje surową telemetrię do okien WINDOW_SEC
    if car.shape[0] == 0:
        return pl.DataFrame()

    return (
        car
        .sort("ts")
        .group_by_dynamic("ts", every=f"{WINDOW_SEC}s", closed="left")
        .agg([
            pl.col("speed").mean().alias("speed_mean"),
            pl.col("speed").std().alias("speed_std"),
            pl.col("speed").max().alias("speed_max"),
            pl.col("rpm").mean().alias("rpm_mean"),
            pl.col("rpm").std().alias("rpm_std"),
            pl.col("throttle").mean().alias("throttle_mean"),
            (pl.col("throttle") > 98).mean().alias("wot_pct"),
            pl.col("brake").mean().alias("brake_mean"),
            pl.col("brake").max().alias("brake_max"),
            pl.col("brake").std().alias("brake_std"),
            pl.col("drs").mean().alias("drs_pct"),
            pl.col("n_gear").mean().alias("gear_mean"),
            pl.col("lap_number").mode().first().alias("lap_number"),
            pl.col("stint_number").mode().first().alias("stint_number"),
            pl.col("compound").mode().first().alias("compound"),
            pl.col("tyre_life").max().alias("tyre_life"),
            pl.len().alias("n_samples"),
        ])
        # Tylko okna z sensowną liczbą próbek (≥3)
        .filter(pl.col("n_samples") >= 3)
    )


def process_driver(
    session_key: int,
    driver_number: int,
    race_control: pl.DataFrame
) -> pl.DataFrame:
    # Pełny pipeline dla jednego kierowcy
    car    = get_car_data(session_key, driver_number)
    laps   = get_laps(session_key, driver_number)
    stints = get_stints(session_key, driver_number)
    pit    = get_pit(session_key, driver_number)

    if car.shape[0] == 0 or laps.shape[0] == 0:
        return pl.DataFrame()

    # Korekta paliwowa
    laps = build_fuel_correction(laps)

    # Złe okrążenia
    bad_laps = get_bad_laps(laps, stints, race_control, pit)

    # Przypisz okrążenia do próbek car_data
    car = assign_lap_to_window(car, laps)

    # Odfiltruj złe okrążenia
    car = car.filter(~pl.col("lap_number").is_in(list(bad_laps)))

    if car.shape[0] == 0:
        return pl.DataFrame()

    # Dodaj compound i tyre_life z stints
    if stints.shape[0] > 0 and "lap_start" in stints.columns:
        stint_map = {}
        for row in stints.iter_rows(named=True):
            for lap in range(row["lap_start"], row.get("lap_end", 99) + 1):
                stint_map[lap] = {
                    "compound":   row.get("compound", "UNKNOWN"),
                    "tyre_life":  lap - row["lap_start"] + 1,
                    "stint_number": row.get("stint_number"),
                }
        car = car.with_columns([
            pl.col("lap_number").map_elements(
                lambda l: stint_map.get(l, {}).get("compound", "UNKNOWN"),
                return_dtype=pl.Utf8
            ).alias("compound"),
            pl.col("lap_number").map_elements(
                lambda l: stint_map.get(l, {}).get("tyre_life", None),
                return_dtype=pl.Int32
            ).alias("tyre_life"),
            pl.col("lap_number").map_elements(
                lambda l: stint_map.get(l, {}).get("stint_number", None),
                return_dtype=pl.Int32
            ).alias("stint_number"),
        ])

    # Okna 10 s
    windows = make_windows(car)
    if windows.shape[0] == 0:
        return pl.DataFrame()

    # Dodaj metadane
    windows = windows.with_columns([
        pl.lit(session_key).alias("session_key"),
        pl.lit(driver_number).alias("driver_number"),
    ])

    # Dołącz lap_time_corr z laps
    lap_cols = laps.select(["lap_number", "lap_time_corr", "fuel_mass_kg"])
    windows = windows.join(lap_cols, on="lap_number", how="left")

    return windows