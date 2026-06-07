import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from config import PROC

def _sanitize_fname(s: str) -> str:
    # Usuń/zmień znaki niebezpieczne do nazwy pliku
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))

def plot_stint(df: pl.DataFrame, session_key: int,
               driver: str, stint: int):
    """
    Wykres jednego stintu z kolorowym tłem wg etykiety:
      zielony=0, żółty=1, czerwony=2
    Oś X: tyre_life` (wiek opony), w przeciwnym razie `lap_number`.
    """
    stint_df = (
        df.filter(
            (pl.col("session_key") == session_key) &
            (pl.col("driver") == driver) &
            (pl.col("stint_number") == stint)
        )
        .sort("lap_number")
    )
    if stint_df.shape[0] == 0:
        print("Brak danych dla tego stintu")
        return

    # Oś X: tyre_life preferowane, fallback na lap_number
    if "tyre_life" in stint_df.columns and stint_df["tyre_life"].drop_nulls().shape[0] > 0:
        x = stint_df["tyre_life"].to_numpy()
        xlabel = "Wiek opony [okrążenia]"
        sort_col = "tyre_life"
    elif "lap_number" in stint_df.columns and stint_df["lap_number"].drop_nulls().shape[0] > 0:
        x = stint_df["lap_number"].to_numpy()
        xlabel = "Okrążenie"
        sort_col = "lap_number"
    else:
        # Ostatecznie indeks okien
        x = np.arange(stint_df.shape[0])
        xlabel = "Okno (indeks)"
        sort_col = None

    # Sortowanie po odpowiedniej kolumnie, jeśli istnieje
    if sort_col:
        stint_df = stint_df.sort(sort_col)
    else:
        # Porządek deterministyczny
        stint_df = stint_df.sort("ts") if "ts" in stint_df.columns else stint_df

    times = stint_df["lap_time_norm_pct"].to_numpy()
    labels = stint_df["target"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = {0: "#d4edda", 1: "#fff3cd", 2: "#f8d7da"}

    for ax, (col, title) in zip(axes, [
        ("lap_time_norm_pct", "Znorm. czas okrążenia [%]"),
        ("brake_energy_proxy", "Proxy energii hamowania"),
        ("speed_std", "Odch. std prędkości [km/h]"),
    ]):
        vals = stint_df[col].to_numpy() if col in stint_df.columns else np.full(x.shape, np.nan)
        ax.plot(x, vals, "k-", linewidth=1.5, zorder=3)

        # Kolorowe tło wg etykiety
        for i in range(len(x) - 1):
            lbl = int(labels[i]) if i < len(labels) else 0
            ax.axvspan(x[i], x[i + 1], color=colors.get(lbl, "#ffffff"), alpha=0.6, zorder=1)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3, zorder=2)

    axes[-1].set_xlabel(xlabel)

    patches = [
        mpatches.Patch(color="#d4edda", label="Stan 0 - brak ryzyka"),
        mpatches.Patch(color="#fff3cd", label="Stan 1 - ryzyko"),
        mpatches.Patch(color="#f8d7da", label="Stan 2 - cliff"),
    ]
    axes[0].legend(handles=patches, loc="upper left", fontsize=8)

    gp_name = stint_df["gp_name"][0] if "gp_name" in stint_df.columns else None

    compound = None
    if "compound" in stint_df.columns and stint_df["compound"].drop_nulls().shape[0] > 0:
        # Dominująca mieszanka w stincie (powinna być jedna)
        compound = stint_df["compound"].drop_nulls().mode()[0]

    gp_part = f" | {gp_name}" if gp_name is not None else ""
    comp_part = f" | compound: {compound}" if compound is not None else ""

    axes[0].set_title(
        f"Stint {stint} | {driver}{gp_part}{comp_part} | session {session_key}",
        fontsize=11
    )

    plt.tight_layout()
    gp_safe = _sanitize_fname(gp_name) if gp_name else "gp"
    comp_safe = _sanitize_fname(compound) if compound else "compound"
    driver_safe = _sanitize_fname(driver)
    fname = PROC / f"stint_{session_key}_{gp_safe}_{driver_safe}_{comp_safe}_stint{stint}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Zapisano: {fname}")
    plt.show()


def plot_example_stints(df: pl.DataFrame, n: int = 3):
    """
    Wybiera do n zróżnicowanych przykładów (różni kierowcy / różne stinty / różne GP)
    i tworzy wykresy dla każdego.
    """
    # Wybieramy unikalne kombinacje (gp_name, session_key, driver, stint_number)
    cols = ["gp_name", "session_key", "driver", "stint_number"]
    available = [c for c in cols if c in df.columns]
    if "stint_number" not in available:
        print("Brak kolumny 'stint_number' — nie można wybrać przykładowych stintów.")
        return

    combos = df.select([c for c in cols if c in df.columns]).drop_nulls().unique().to_dicts()
    if not combos:
        print("Brak dostępnych stintów do wygenerowania przykładów.")
        return

    # Preferuj różnorodność: próbuj zebrać unikalnych kierowców/GP
    chosen = []
    drivers_seen = set()
    gps_seen = set()
    for rec in combos:
        driver = rec.get("driver")
        gp = rec.get("gp_name")
        stint = rec.get("stint_number")
        session_key = rec.get("session_key")
        key = (driver, gp, stint, session_key)
        if driver is None or stint is None:
            continue
        # preferuj nowe kierowcy i nowe gp
        score = (driver not in drivers_seen) + (gp not in gps_seen if gp is not None else 0)
        if score > 0 or len(chosen) < n:
            chosen.append(key)
            drivers_seen.add(driver)
            if gp is not None:
                gps_seen.add(gp)
        if len(chosen) >= n:
            break

    # Jeśli wybrano mniej niż n, dopełnij kolejnymi
    if len(chosen) < n:
        for rec in combos:
            driver = rec.get("driver"); gp = rec.get("gp_name"); stint = rec.get("stint_number"); session_key = rec.get("session_key")
            key = (driver, gp, stint, session_key)
            if key not in chosen and driver is not None and stint is not None:
                chosen.append(key)
            if len(chosen) >= n:
                break

    # Rysuj każdy wybrany
    for driver, gp, stint, session_key in chosen[:n]:
        try:
            plot_stint(df, session_key=int(session_key), driver=str(driver), stint=int(stint))
        except Exception as e:
            print(f"Nie udało się narysować {driver} stint {stint} (session {session_key}): {e}")


def plot_class_distribution(df: pl.DataFrame):
    # Rozkład klas per compound
    counts = (
        df.group_by(["compound", "target"])
          .agg(pl.len().alias("n"))
          .sort(["compound", "target"])
    )
    compounds = counts["compound"].unique().sort().to_list()
    targets = [0, 1, 2]
    x = np.arange(len(compounds))
    width = 0.25
    clrs = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(targets):
        vals = []
        for c in compounds:
            row = counts.filter(
                (pl.col("compound") == c) & (pl.col("target") == t)
            )
            vals.append(row["n"][0] if row.shape[0] > 0 else 0)
        ax.bar(x + i * width, vals, width, label=f"Stan {t}", color=clrs[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(compounds)
    ax.set_xlabel("Mieszanka")
    ax.set_ylabel("Liczba okien")
    ax.set_title("Rozkład klas per mieszanka")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROC / "class_distribution.png", dpi=150)
    plt.show()


def plot_correlation_matrix(df: pl.DataFrame, feature_cols: list[str]):
    # Macierz korelacji dla wybranych cech
    # Filtruj tylko kolumny numeryczne
    numeric_cols = [c for c in feature_cols if c in df.columns]
    if not numeric_cols:
        print("Brak kolumn numerycznych do obliczenia korelacji")
        return

    subset = df.select(numeric_cols[:20]).to_pandas() # top 20 cech
    # Upewnij się, że wszystkie kolumny są numeryczne
    subset = subset.select_dtypes(include=[np.number])

    if subset.shape[1] < 2:
        print("Za mało kolumn numerycznych do obliczenia korelacji")
        return
    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Macierz korelacji cech")
    plt.tight_layout()
    plt.savefig(PROC / "correlation_matrix.png", dpi=150)
    plt.show()


def plot_degradation_by_compound(df):
    # Średni lap_time_norm_pct vs tyre_life per mieszanka (SOFT/MEDIUM/HARD)
    fig, ax = plt.subplots(figsize=(10, 5))
    for compound, color in [("SOFT","#e74c3c"), ("MEDIUM","#f39c12"), ("HARD","#95a5a6")]:
        sub = df.filter(pl.col("compound") == compound)
        grouped = sub.group_by("tyre_life").agg(
            pl.col("lap_time_norm_pct").mean()
        ).sort("tyre_life")
        ax.plot(grouped["tyre_life"], grouped["lap_time_norm_pct"],
                label=compound, color=color, linewidth=2)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="próg Stan 1")
    ax.axhline(1.5, linestyle="--", color="red", alpha=0.5, label="próg Stan 2")
    ax.set_xlabel("Wiek opony [okrążenia]")
    ax.set_ylabel("Znorm. czas okrążenia [%]")
    ax.set_title("Degradacja per mieszanka")
    ax.legend(); plt.tight_layout()
    plt.savefig(PROC / "degradation_by_compound.png", dpi=150)


if __name__ == "__main__":
    from features import get_feature_columns

    df = pl.read_parquet(PROC / "features_all.parquet")
    print(f"Załadowano: {df.shape}")
    print(f"Klasy: {df['target'].value_counts().sort('target')}")

    # Znajdź pary cech |korelacja| > 0.95
    feature_cols = get_feature_columns(df)
    feat_df = df.select([c for c in feature_cols if c in df.columns]).to_pandas()
    feat_df = feat_df.select_dtypes(include=[np.number])
    if feat_df.shape[1] < 2:
        print("Za mało numerycznych cech do analizy korelacji.")
    else:
        corr = feat_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        print("Do usunięcia:", to_drop)

    # Przykładowy stint
    sample = df.filter(pl.col("target").is_in([1, 2])).head(1)
    if sample.shape[0] > 0:
        plot_stint(
            df,
            session_key=int(sample["session_key"][0]),
            driver=str(sample["driver"][0]),
            stint=int(sample["stint_number"][0])
        )

    plot_example_stints(df)
    plot_class_distribution(df)
    plot_correlation_matrix(df, get_feature_columns(df))
    plot_degradation_by_compound(df)