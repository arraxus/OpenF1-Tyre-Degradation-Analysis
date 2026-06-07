import polars as pl
from config import CLIFF_THRESHOLD_PCT, RISK_THRESHOLD_PCT, BRAKE_ENERGY_WINDOW, SPEED_VAR_WINDOW, LAG_STEPS, METADATA_EXCLUDE

def add_lap_time_norm(df: pl.DataFrame) -> pl.DataFrame:
    # Znormalizowany czas okrążenia względem mediany stintu
    # Podstawa i powód wycieku danych- model uczy się progów
    df = df.with_columns(
        pl.col("lap_time_corr")
          .median()
          .over(["session_key", "driver_number", "stint_number"])
          .alias("stint_median_lap")
    )
    df = df.with_columns(
        ((pl.col("lap_time_corr") - pl.col("stint_median_lap"))
         / pl.col("stint_median_lap") * 100)
        .alias("lap_time_norm_pct")
    )
    return df


def add_quali_delta(df: pl.DataFrame) -> pl.DataFrame:
    """Delta czasu okrążenia względem czasu kwalifikacji kierowcy [%].
    O ile % wolniej jedzie kierowca w wyścigu względem
    swojego czasu kwalifikacji na tym samym torze.
    Nie zależna od bieżącego stintu i rośnie monotonicznie z wiekiem opony
    """
    if "quali_best_lap_s" not in df.columns:
        # Kwalifikacje nie zostały pobrane — dodaj pustą kolumnę
        return df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("lap_time_quali_delta_pct")
        )

    df = df.with_columns(
        ((pl.col("lap_time_corr") - pl.col("quali_best_lap_s"))
         / pl.col("quali_best_lap_s") * 100)
        .alias("lap_time_quali_delta_pct")
    )
    return df


def add_proxy_temperature(df: pl.DataFrame) -> pl.DataFrame:
    # Proxy temperatury opony przez energię hamowania
    df = df.sort("ts")
    df = df.with_columns([
        # Kumulatywna energia hamowania: Brake^2, rolling 60s
        (pl.col("brake_mean") ** 2)
          .rolling_mean(window_size=6)  # ~60s przy 10s oknach
          .over(["session_key", "driver_number", "stint_number"])
          .alias("brake_energy_proxy"),
    ])
    return df


def add_lag_features(df: pl.DataFrame, n_lags: int = LAG_STEPS) -> pl.DataFrame:
    # Lag features — pamięć krótkoterminowa modelu
    lag_cols = ["speed_mean", "speed_std", "brake_max", "wot_pct",
                "rpm_std", "throttle_mean", "lap_time_norm_pct",
                "lap_time_quali_delta_pct", "brake_energy_proxy"]

    exprs = []
    for col in lag_cols:
        if col in df.columns:
            for lag in range(1, n_lags + 1):
                exprs.append(
                    pl.col(col)
                      .shift(lag)
                      .over(["session_key", "driver_number", "stint_number"])
                      .alias(f"{col}_lag{lag}")
                )
    return df.with_columns(exprs)


def add_rolling_stats(df: pl.DataFrame) -> pl.DataFrame:
    # Rolling statistics — trend degradacji
    df = df.with_columns([
        # Trend czasu okrążenia: średnia krocząca z 5 okien (~50s)
        pl.col("lap_time_norm_pct")
          .rolling_mean(window_size=5)
          .over(["session_key", "driver_number", "stint_number"])
          .alias("lap_time_rolling_mean"),

        # Przyspieszenie degradacji (różnica rolling means)
        (pl.col("lap_time_norm_pct").rolling_mean(window_size=3)
         - pl.col("lap_time_norm_pct").rolling_mean(window_size=6))
          .over(["session_key", "driver_number", "stint_number"])
          .alias("degradation_rate"),

        # Max hamulec w ostatnich 5 oknach
        pl.col("brake_max")
          .rolling_max(window_size=5)
          .over(["session_key", "driver_number", "stint_number"])
          .alias("brake_max_rolling"),

        # Trend delty względem kwalifikacji
        (pl.col("lap_time_quali_delta_pct").rolling_mean(window_size=3)
         - pl.col("lap_time_quali_delta_pct").rolling_mean(window_size=6))
        .over(["session_key", "driver_number", "stint_number"])
        .alias("quali_delta_degradation_rate"),
    ])
    return df


def assign_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Etykiety:
      0 — brak ryzyka (stabilna opona)
      1 — ryzyko umiarkowane (początek degradacji)
      2 — cliff (krytyczny spadek przyczepności)
    Docelowo etykiety generowane PELT z ruptures.
    """
    df = df.with_columns(
        pl.when(pl.col("lap_time_norm_pct") > CLIFF_THRESHOLD_PCT)
          .then(2)
          .when(
              (pl.col("lap_time_norm_pct") > RISK_THRESHOLD_PCT) |
              (pl.col("degradation_rate") > 0.3) |
              (pl.col("brake_energy_proxy") > pl.col("brake_energy_proxy")
                 .mean().over(["session_key", "driver_number", "stint_number"]) * 1.5)
          )
          .then(1)
          .otherwise(0)
          .cast(pl.Int8)
          .alias("target")
    )
    return df


def build_feature_matrix(df: pl.DataFrame) -> pl.DataFrame:
    # Pełny pipeline inżynierii cech
    required_input = ["lap_time_corr", "session_key", "driver_number", "stint_number"]
    missing = [c for c in required_input if c not in df.columns]
    if missing:
        raise ValueError(f"Brak wymaganych kolumn wejściowych: {missing}")

    df = add_lap_time_norm(df)
    df = add_quali_delta(df)
    df = add_proxy_temperature(df)
    df = add_rolling_stats(df)
    df = add_lag_features(df)
    df = assign_labels(df)

    # Usuń okna z brakami w kluczowych kolumnach (pierwsze lagi)
    required = ["lap_time_norm_pct", "brake_energy_proxy",
                "speed_mean_lag1", "brake_max_lag1", "target"]
    existing = [c for c in required if c in df.columns]
    df = df.drop_nulls(subset=existing)

    # Raport pokrycia kwalifikacji
    if "lap_time_quali_delta_pct" in df.columns:
        n_total = df.shape[0]
        n_filled = df["lap_time_quali_delta_pct"].drop_nulls().shape[0]
        print(f"  quali_delta: {n_filled}/{n_total} okien z danymi "
              f"({100 * n_filled / n_total:.1f}%)")

    return df


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    # Zwraca listę kolumn-cech (bez metadanych i target).
    return [c for c in df.columns if c not in METADATA_EXCLUDE]