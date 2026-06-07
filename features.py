import numpy as np
import ruptures as rpt
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


def assign_labels_pelt(
    df: pl.DataFrame,
    signal_col: str = "degradation_rate",
    group_cols: tuple[str, str, str] = ("session_key", "driver_number", "stint_number"),
    n_bkps: int = 2,
    model: str = "rbf",
    min_size: int = 3,
    jump: int = 1,
    smooth_window: int = 3,           # wygładzenie rolling median przed PELT
    min_mean_diff: float = 0.35,      # minimalna absolutna różnica mean(segment_max) - mean(prev)
    min_relative_diff: float = 1.3,   # minimalny względny ratio mean_max / mean_prev
    max_cliff_frac: float = 0.4,      # jeśli cliff segment > 40% stintu i nie ma skoku -> nie cliff
    max_cliff_len: int | None = None, # opcjonalny bezwzględny limit długości cliff (liczba okien)
    jump_threshold: float = 0.5,      # jeśli maksymalny jednopunktowy skok < threshold -> brak cliff
) -> pl.DataFrame:
    """
    Rozszerzona wersja PELT z dodatkowymi regułami, by uniknąć nadmiernego przypisywania Stanu 2.

    Zwraca wersję dataframe z kolumną 'target' (Int8).
    """
    if signal_col not in df.columns:
        raise ValueError(f"Brak kolumny sygnału dla PELT: {signal_col}")

    if rpt is None:
        print("  [WARN] Brak pakietu ruptures, fallback do assign_labels()")
        return assign_labels(df)

    sort_cols = [c for c in ("session_key", "driver_number", "stint_number", "ts") if c in df.columns]
    if sort_cols:
        df_work = df.sort(sort_cols)
    else:
        df_work = df

    df_work = df_work.with_row_count("row_id")
    out_parts = []

    for g in df_work.partition_by(list(group_cols), maintain_order=True):
        row_ids = g["row_id"].to_numpy()
        x = g[signal_col].cast(pl.Float64).to_numpy()

        # Clean NaN/inf
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        if not finite.all():
            idx = np.arange(len(x))
            if finite.any():
                x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
            else:
                x[:] = 0.0

        n = len(x)
        # Smoothing krótkie okno (median) — usuwa szum
        if smooth_window is not None and smooth_window > 1 and n >= smooth_window:
            from scipy.signal import medfilt
            try:
                x_s = medfilt(x, kernel_size=smooth_window)
            except Exception:
                # fallback na prostą konwolucję jeśli medfilt niedostępny
                k = np.ones(smooth_window) / smooth_window
                x_s = np.convolve(x, k, mode="same")
        else:
            x_s = x.copy()

        # Fallback dla krótkich/płaskich sygnałów
        if n < max(8, (n_bkps + 1) * min_size) or np.nanstd(x_s) < 1e-9:
            pos = np.arange(n)
            q1, q2 = np.quantile(pos, [1/3, 2/3]) if n > 2 else (0, 1)
            target = np.where(pos <= q1, 0, np.where(pos <= q2, 1, 2)).astype(np.int8)
            out_parts.append(pl.DataFrame({"row_id": row_ids, "target": target}))
            continue

        # Opcjonalna szybka detekcja skoku jednopunktowego
        max_jump = np.max(np.abs(np.diff(x_s))) if n > 1 else 0.0

        signal = x_s.reshape(-1, 1)
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)

        penalties = np.logspace(-3, 2, 30) * max(np.nanvar(x_s), 1e-6)
        best_bkps = [n]
        best_diff = 10**9
        for pen in penalties:
            bkps = algo.predict(pen=float(pen))
            k = max(0, len(bkps) - 1)
            diff = abs(k - n_bkps)
            if diff < best_diff:
                best_diff = diff
                best_bkps = bkps
            if diff == 0:
                break

        # segment ids
        seg_id = np.zeros(n, dtype=int)
        start = 0
        for sid, end in enumerate(best_bkps):
            seg_id[start:end] = sid
            start = end

        uniq = np.unique(seg_id)
        # compute segment means and lengths
        seg_means = []
        seg_lens = []
        for u in uniq:
            mask = seg_id == u
            seg_means.append(np.nanmean(x_s[mask]) if mask.any() else 0.0)
            seg_lens.append(mask.sum())
        seg_means = np.array(seg_means)
        seg_lens = np.array(seg_lens)

        # identify candidate cliff segment: the one with max mean
        max_idx = int(np.argmax(seg_means))
        max_mean = seg_means[max_idx]
        prev_mean = seg_means[max_idx - 1] if max_idx > 0 else np.nan
        prev_mean = float(prev_mean) if not np.isnan(prev_mean) else np.nan

        # default mapping: map segments by rank -> 0..2 (cap to 2)
        if len(uniq) == 1:
            target = np.zeros(n, dtype=np.int8)
        else:
            # map segments to provisional states by ranking their mean (lowest->0, mid->1, highest->2)
            ranks = np.argsort(np.argsort(seg_means))  # gives rank per segment: 0..k-1
            # scale ranks into 0..2
            if ranks.max() > 0:
                scaled = (ranks / ranks.max() * 2.0).round().astype(int)
            else:
                scaled = ranks.astype(int)
            # build target from segment mapping
            seg_to_state = {u: int(scaled[i]) if i < len(scaled) else 1 for i, u in enumerate(uniq)}
            target = np.array([seg_to_state[s] for s in seg_id], dtype=np.int8)

        # Post-processing: require that highest-mean segment satisfies jump/mean-diff/length rules to be cliff
        candidate_mask = seg_id == uniq[max_idx]
        candidate_len = seg_lens[max_idx]
        candidate_frac = candidate_len / max(n, 1)

        is_cliff = False
        # check absolute mean diff if previous exists
        if not np.isnan(prev_mean):
            abs_diff_ok = (max_mean - prev_mean) >= min_mean_diff
            rel_diff_ok = (prev_mean == 0 and max_mean > 0) or (prev_mean > 0 and (max_mean / prev_mean) >= min_relative_diff)
        else:
            abs_diff_ok = max_mean >= min_mean_diff
            rel_diff_ok = max_mean >= min_mean_diff

        jump_ok = max_jump >= jump_threshold

        # Cliff only if one of differences is satisfied AND (either jump_ok OR length small enough)
        if (abs_diff_ok or rel_diff_ok) and (jump_ok or (candidate_frac <= max_cliff_frac and (max_cliff_len is None or candidate_len <= max_cliff_len))):
            is_cliff = True

        if not is_cliff:
            # downgrade candidate-2 to 1 (ryzyko) if it was assigned 2
            target[candidate_mask] = np.where(target[candidate_mask] == 2, 1, target[candidate_mask]).astype(np.int8)

        # Safety: ensure labels in {0,1,2}
        target = np.clip(target, 0, 2).astype(np.int8)
        out_parts.append(pl.DataFrame({"row_id": row_ids, "target": target}))

    labels_df = pl.concat(out_parts, how="vertical")
    df_labeled = df_work.join(labels_df, on="row_id", how="left").drop("row_id")
    df_labeled = df_labeled.with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
    return df_labeled


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
    #df = assign_labels(df)

    # Etykiety przez PELT (bez leakage z progów lap_time_norm_pct)
    df = assign_labels_pelt(df, signal_col="degradation_rate")

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