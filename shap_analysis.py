import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.metrics import log_loss, brier_score_loss, f1_score
from features import METADATA_EXCLUDE
from eda import _sanitize_fname
from models import prepare_data, chronological_split, sort_df
from config import PROC


# 1. SHAP — globalna wyjaśnialność

def run_shap_analysis(model, X, y, feature_cols, sample_n=3000):
    rng = np.random.default_rng(42)
    idx_per_class = []
    for cls in [0, 1, 2]:
        idx_cls = np.where(y == cls)[0]
        n = min(len(idx_cls), sample_n // 3)
        idx_per_class.append(rng.choice(idx_cls, n, replace=False))
    sample_idx = np.concatenate(idx_per_class)
    rng.shuffle(sample_idx)

    X_sample = X[sample_idx]
    y_sample = y[sample_idx]

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    if not isinstance(raw, np.ndarray) or raw.ndim != 3:
        raise ValueError(
            f"Nieoczekiwany format SHAP values: typ={type(raw)}, "
            f"ndim={getattr(raw, 'ndim', None)}. "
            "Oczekiwano ndarray o kształcie (n_samples, n_features, n_classes)."
        )

    # Lista per klasa: każdy element ma shape (n_samples, n_features)
    shap_values = [raw[:, :, i] for i in range(raw.shape[2])]

    print(f"shap_values: {len(shap_values)} klas, shape: {shap_values[0].shape}")
    return explainer, shap_values, X_sample, y_sample, sample_idx


def plot_shap_beeswarm(shap_values, X_sample, feature_cols,
                       class_idx: int, class_name: str):
    # Beeswarm plot dla jednej klasy
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, class_idx, :]
    else:
        sv = shap_values

    # Bezpieczne sprawdzenie wymiarów
    if sv.shape[1] != X_sample.shape[1]:
        print(f"UWAGA: SHAP values ma {sv.shape[1]} cech, X_sample ma {X_sample.shape[1]}")
        print(f"       Używam tylko pierwszych {min(sv.shape[1], X_sample.shape[1])} cech")
        # Spróbuj przyciąć do wspólnego rozmiaru
        n_feat = min(sv.shape[1], X_sample.shape[1])
        sv = sv[:, :n_feat]
        X_plot = X_sample.iloc[:, :n_feat]
        feat_plot = feature_cols[:n_feat]
    else:
        X_plot = X_sample
        feat_plot = feature_cols

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_plot,
        feature_names=feat_plot,
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.title(f"SHAP — {class_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(PROC / f"shap_beeswarm_{class_name.replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar_all_classes(shap_values, feature_cols):
    # Porównanie mean |SHAP| dla wszystkich 3 klas obok siebie
    class_names = ["Stan 0", "Stan 1", "Stan 2"]
    means = [
        np.abs(sv).mean(axis=0) for sv in shap_values
    ]

    # Top 15 cech wg sumy ważności across klas
    total = sum(means)
    top_idx = np.argsort(total)[-15:][::-1]
    top_names = [feature_cols[i] for i in top_idx]

    x = np.arange(len(top_names))
    width = 0.25
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mean, name, color) in enumerate(zip(means, class_names, colors)):
        ax.bar(x + i * width, mean[top_idx], width,
               label=name, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Ważność cech per klasa — SHAP")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROC / "shap_bar_all_classes.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# 2. KLUCZOWY EKSPERYMENT — czy model przewiduje cliff PRZED jego wystąpieniem?

def experiment_no_lap_time(df, feature_cols, splits):
    """
    Trenuje XGBoost BEZ lap_time_norm_pct i jego lagów.
    Sprawdza czy model nadal wykrywa cliff na podstawie
    czystych sygnałów telemetrycznych.
    Odpowiada na pytanie badawcze.
    """
    print("\n" + "═" * 60)
    print("EKSPERYMENT — model XGBoost bez lap_time_norm_pct")
    print("Pytanie: czy sygnały telemetryczne wystarczą do")
    print("przewidzenia cliffu BEZ znajomości czasu okrążenia?")
    print("═" * 60)

    from sklearn.metrics import f1_score, classification_report
    import xgboost as xgb

    # Usuń lap_time i jego pochodne
    lap_time_cols = [c for c in feature_cols if "lap_time" in c]
    print(f"\nUsuwam kolumny: {lap_time_cols}")

    tel_cols = [c for c in feature_cols if "lap_time" not in c]
    print(f"Pozostałe cechy ({len(tel_cols)}): {tel_cols[:5]}...")

    exclude_base = set(METADATA_EXCLUDE)
    exclude_base.add("lap_time_rolling_mean")
    exclude_base.update(lap_time_cols)

    df_sorted = sort_df(df)

    group_col = "race_date" if "race_date" in df_sorted.columns else "session_key"
    groups_tel = df_sorted[group_col].to_numpy()
    X_tel = df_sorted.select(tel_cols).to_numpy().astype(np.float32)
    y = df_sorted["target"].to_numpy().astype(np.int32)

    splits_tel = chronological_split(X_tel, y, groups_tel, n_splits=3)

    train_idx, test_idx = splits_tel[-1]
    X_tr, y_tr = X_tel[train_idx], y[train_idx]
    X_te, y_te = X_tel[test_idx],  y[test_idx]

    class_counts = np.bincount(y_tr)
    weights = np.array([len(y_tr) / (3 * class_counts[c]) for c in y_tr])

    model_tel = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    model_tel.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
    y_pred = model_tel.predict(X_te)

    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\nF1-macro (bez lap_time): {f1:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Stan0","Stan1","Stan2"],
                                zero_division=0))

    return model_tel, f1, tel_cols


# 3. PORÓWANIE XGBOOST — przed i po kalibracji

def compare_calibration(model_uncal, model_cal, X_test, y_test, feature_cols):
    """
    Porównuje model oryginalny vs skalibrowany.
    Pokazuje metryki kalibracji i wizualizacje.
    """
    print("\n" + "═" * 70)
    print("RAPORT PORÓWNAWCZY — Kalibracja prawdopodobieństw")
    print("═" * 70)

    # Predykcje
    y_pred_uncal = model_uncal.predict(X_test)
    y_pred_cal = model_cal.predict(X_test)

    y_proba_uncal = model_uncal.predict_proba(X_test)
    y_proba_cal = model_cal.predict_proba(X_test)

    # Metryki
    f1_uncal = f1_score(y_test, y_pred_uncal, average="macro", zero_division=0)
    f1_cal = f1_score(y_test, y_pred_cal, average="macro", zero_division=0)

    log_loss_uncal = log_loss(y_test, y_proba_uncal)
    log_loss_cal = log_loss(y_test, y_proba_cal)

    # Brier Score dla wszystkich 3 klas (uśredniony)
    brier_uncal_list = []
    brier_cal_list = []
    for k in range(3):
        y_bin = (y_test == k).astype(int)
        brier_uncal_list.append(brier_score_loss(y_bin, y_proba_uncal[:, k]))
        brier_cal_list.append(brier_score_loss(y_bin, y_proba_cal[:, k]))

    brier_uncal = np.mean(brier_uncal_list)
    brier_cal = np.mean(brier_cal_list)

    print(f"\n📊 METRYKI:")
    print(f"{'Metrika':<25} {'Przed kalibracją':>20} {'Po kalibracji':>20} {'Zmiana':>15}")
    print("-" * 80)
    print(f"{'F1-macro':<25} {f1_uncal:>20.4f} {f1_cal:>20.4f} {f1_cal - f1_uncal:>+15.4f}")
    print(f"{'Log Loss':<25} {log_loss_uncal:>20.4f} {log_loss_cal:>20.4f} {log_loss_cal - log_loss_uncal:>+15.4f}")
    print(f"{'Brier Score (avg)':<25} {brier_uncal:>20.4f} {brier_cal:>20.4f} {brier_cal - brier_uncal:>+15.4f}")

    # Per-class Brier
    print(f"\nBrier Score per klasa:")
    class_names = ["Stan 0", "Stan 1", "Stan 2"]
    for k, name in enumerate(class_names):
        print(f"  {name:<15} przed: {brier_uncal_list[k]:.4f}  |  po: {brier_cal_list[k]:.4f}  |  "
              f"zmiana: {brier_cal_list[k] - brier_uncal_list[k]:+.4f}")

    # Analiza per klasa — extremes prawdopodobieństw
    print(f"\n📈 ROZKŁAD PRAWDOPODOBIEŃSTW (klasa Stan 2):")
    print(f"{'Statystyka':<25} {'Przed kalibracją':>20} {'Po kalibracji':>20}")
    print("-" * 65)

    p_uncal_c2 = y_proba_uncal[:, 2]
    p_cal_c2 = y_proba_cal[:, 2]

    print(f"{'Min':<25} {np.min(p_uncal_c2):>20.4f} {np.min(p_cal_c2):>20.4f}")
    print(f"{'Q1 (25%)':<25} {np.percentile(p_uncal_c2, 25):>20.4f} {np.percentile(p_cal_c2, 25):>20.4f}")
    print(f"{'Mediana':<25} {np.median(p_uncal_c2):>20.4f} {np.median(p_cal_c2):>20.4f}")
    print(f"{'Q3 (75%)':<25} {np.percentile(p_uncal_c2, 75):>20.4f} {np.percentile(p_cal_c2, 75):>20.4f}")
    print(f"{'Max':<25} {np.max(p_uncal_c2):>20.4f} {np.max(p_cal_c2):>20.4f}")
    print(f"{'Std dev':<25} {np.std(p_uncal_c2):>20.4f} {np.std(p_cal_c2):>20.4f}")

    # Liczba "pewnych" predykcji (P > 0.9 lub P < 0.1)
    extreme_uncal = np.sum((p_uncal_c2 > 0.9) | (p_uncal_c2 < 0.1))
    extreme_cal = np.sum((p_cal_c2 > 0.9) | (p_cal_c2 < 0.1))

    print(f"\n🎯 PEWNE PREDYKCJE (P(Stan 2) > 0.9 lub < 0.1):")
    print(f"{'Przed kalibracją':<25} {extreme_uncal:>20} ({100 * extreme_uncal / len(y_test):.1f}%)")
    print(f"{'Po kalibracji':<25} {extreme_cal:>20} ({100 * extreme_cal / len(y_test):.1f}%)")

    # Wizualizacja — rozkład prawdopodobieństw
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(p_uncal_c2, bins=30, alpha=0.7, color="#d62728", edgecolor="black")
    axes[0].set_xlabel("P(Stan 2)")
    axes[0].set_ylabel("Liczba próbek")
    axes[0].set_title("Rozkład — Przed kalibracją")
    axes[0].grid(alpha=0.3)

    axes[1].hist(p_cal_c2, bins=30, alpha=0.7, color="#1f77b4", edgecolor="black")
    axes[1].set_xlabel("P(Stan 2)")
    axes[1].set_ylabel("Liczba próbek")
    axes[1].set_title("Rozkład — Po kalibracji")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PROC / "calibration_comparison_hist.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Violin plot porównanie
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(
        [p_uncal_c2, p_cal_c2],
        positions=[1, 2],
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Przed kalibracją", "Po kalibracji"])
    ax.set_ylabel("P(Stan 2)")
    ax.set_title("Rozkład prawdopodobieństw — klasa Stan 2")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PROC / "calibration_comparison_violin.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "f1_uncal": f1_uncal,
        "f1_cal": f1_cal,
        "log_loss_uncal": log_loss_uncal,
        "log_loss_cal": log_loss_cal,
        "brier_uncal": brier_uncal,
        "brier_cal": brier_cal,
    }


# 4. LIVE FEED — symulacja predykcji w czasie rzeczywistym

def live_feed_simulation(model, df: pl.DataFrame,
                         feature_cols: list,
                         session_key: int, driver: str,
                         stint_number: int):
    """
    Symuluje predykcję okno po oknie dla jednego stintu.
    Odtwarza dane sekwencyjnie i pokazuje jak model
    klasyfikuje stan opony w 'czasie rzeczywistym'.
    """
    print(f"\n{'═'*60}")
    print(f"LIVE FEED: {driver} | session {session_key} | stint {stint_number}")
    print("═" * 60)

    stint_df = (
        df.filter(
            (pl.col("session_key")   == session_key) &
            (pl.col("driver")        == driver) &
            (pl.col("stint_number")  == stint_number)
        )
        .sort("tyre_life")
    )

    if stint_df.shape[0] == 0:
        print("Brak danych dla tego stintu!")
        return

    # Tylko dostępne cechy
    avail = [c for c in feature_cols if c in stint_df.columns]
    X_stint = stint_df.select(avail).to_numpy().astype(np.float32)
    y_true  = stint_df["target"].to_numpy()

    # Uzupełnij NaN medianą kolumny (dla brakujących lagów)
    col_medians = np.nanmedian(X_stint, axis=0)
    for j in range(X_stint.shape[1]):
        mask = np.isnan(X_stint[:, j])
        X_stint[mask, j] = col_medians[j]

    state_names   = ["✅ Stan 0 (OK)", "⚠️  Stan 1 (ryzyko)", "🔴 Stan 2 (CLIFF)"]
    state_colors  = ["green", "orange", "red"]
    state_history = []
    prob_history  = []

    tyre_ages = stint_df["tyre_life"].to_numpy() if "tyre_life" in stint_df.columns \
                else np.arange(len(y_true))
    compound  = stint_df["compound"][0] if "compound" in stint_df.columns else "?"
    gp_name = stint_df["gp_name"][0] if "gp_name" in stint_df.columns else "?"

    print(f"\n{'Wiek':>5} | {'Predykcja':<22} | {'Prawdopodobieństwa':>35} | {'Prawda'}")
    print("-" * 80)

    for i in range(X_stint.shape[0]):
        x_row = X_stint[i:i+1]
        proba = model.predict_proba(x_row)[0]
        pred  = int(np.argmax(proba))
        true  = int(y_true[i])
        age   = int(tyre_ages[i]) if i < len(tyre_ages) else i

        state_history.append(pred)
        prob_history.append(proba)

        marker = "✓" if pred == true else "✗"
        print(
            f"{age:>5} | {state_names[pred]:<22} | "
            f"[{proba[0]:.2f}, {proba[1]:.2f}, {proba[2]:.2f}] | "
            f"{state_names[true]} {marker}"
        )

    # Wykres live feed
    _plot_live_feed(
        tyre_ages, state_history, prob_history, y_true,
        driver, session_key, stint_number, compound, gp_name
    )

    # Podsumowanie
    correct = sum(p == t for p, t in zip(state_history, y_true))
    print(f"\nDokładność na stincie: {correct}/{len(y_true)} "
          f"({100*correct/len(y_true):.1f}%)")

    # Detekcja cliffu — kiedy model po raz pierwszy dał Stan 2?
    cliff_pred = next((i for i, s in enumerate(state_history) if s == 2), None)
    cliff_true = next((i for i, s in enumerate(y_true) if s == 2), None)

    if cliff_true is not None:
        if cliff_pred is not None:
            lead = cliff_true - cliff_pred
            age_pred = int(tyre_ages[cliff_pred]) if cliff_pred < len(tyre_ages) else cliff_pred
            age_true = int(tyre_ages[cliff_true]) if cliff_true < len(tyre_ages) else cliff_true
            print(f"\nCliff wykryty przez model: okrążenie ~{age_pred} "
                  f"(prawdziwy: ~{age_true})")
            if lead > 0:
                print(f"✅ Model wyprzedził cliff o {lead} okien (~{lead*10}s)")
            elif lead == 0:
                print("⚠️  Model wykrył cliff dokładnie w momencie wystąpienia")
            else:
                print(f"❌ Model spóźnił się o {-lead} okien (~{-lead*10}s)")
        else:
            print("\n❌ Model nie wykrył cliffu w tym stincie")
    else:
        print("\nℹ️  Brak cliffu w tym stincie (Stan 2 nie wystąpił)")

    return state_history, prob_history


def _plot_live_feed(tyre_ages, state_history, prob_history, y_true,
                    driver, session_key, stint_number, compound, gp_name):
    # Wykres prawdopodobieństw i predykcji dla live feed
    prob_arr = np.array(prob_history)

    gp_safe = _sanitize_fname(str(gp_name))
    driver_safe = _sanitize_fname(str(driver))

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

    # Górny panel — prawdopodobieństwa
    ax1 = fig.add_subplot(gs[0])
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["P(Stan 0)", "P(Stan 1)", "P(Stan 2 — cliff)"]
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        ax1.plot(tyre_ages[:len(prob_arr)], prob_arr[:, i],
                 color=col, linewidth=2, label=lbl)

    # Zaznacz próg decyzyjny
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Prawdopodobieństwo")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(
        f"Live Feed | {driver} | {gp_name} | stint {stint_number} | "
        f"{compound} | session {session_key}",
        fontsize=11
    )
    ax1.set_xticklabels([])

    # Dolny panel — predykcja vs prawda
    ax2 = fig.add_subplot(gs[1])
    pred_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    for i, age in enumerate(tyre_ages[:len(state_history)]):
        ax2.barh(0.3, 1, left=age - 0.5,
                 color=pred_colors[state_history[i]], alpha=0.7, height=0.4)
        ax2.barh(-0.3, 1, left=age - 0.5,
                 color=pred_colors[int(y_true[i])], alpha=0.7, height=0.4)

    ax2.set_xlim(tyre_ages[0] - 1, tyre_ages[-1] + 1)
    ax2.set_yticks([0.3, -0.3])
    ax2.set_yticklabels(["Predykcja", "Prawda"], fontsize=8)
    ax2.set_xlabel("Wiek opony [okrążenia]")
    ax2.set_ylim(-0.7, 0.7)

    plt.tight_layout()
    out_name = PROC / f"livefeed_{session_key}_{gp_safe}_{driver_safe}_stint{stint_number}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Zapisano: {out_name}")
    plt.show()


def live_feed_comparison(model_uncal, model_cal, df: pl.DataFrame,
                         feature_cols: list,
                         session_key: int, driver: str,
                         stint_number: int):
    """
    Pokazuje live feed dla modelu przed i po kalibracji
    na jednym wykresie (dwie krzywe P(Stan 2)).
    """
    print(f"\n{'─' * 70}")
    print(f"LIVE FEED PORÓWNANIE: {driver} | session {session_key} | stint {stint_number}")
    print("─" * 70)

    stint_df = (
        df.filter(
            (pl.col("session_key") == session_key) &
            (pl.col("driver") == driver) &
            (pl.col("stint_number") == stint_number)
        )
        .sort("tyre_life")
    )

    if stint_df.shape[0] == 0:
        print("❌ Brak danych dla tego stintu!")
        return

    avail = [c for c in feature_cols if c in stint_df.columns]
    X_stint = stint_df.select(avail).to_numpy().astype(np.float32)
    y_true = stint_df["target"].to_numpy()

    # Uzupełnij NaN medianą
    col_medians = np.nanmedian(X_stint, axis=0)
    for j in range(X_stint.shape[1]):
        mask = np.isnan(X_stint[:, j])
        X_stint[mask, j] = col_medians[j]

    # Predykcje
    proba_uncal = model_uncal.predict_proba(X_stint)
    proba_cal = model_cal.predict_proba(X_stint)

    tyre_ages = stint_df["tyre_life"].to_numpy() if "tyre_life" in stint_df.columns \
        else np.arange(len(y_true))
    compound = stint_df["compound"][0] if "compound" in stint_df.columns else "?"
    gp_name = stint_df["gp_name"][0] if "gp_name" in stint_df.columns else "?"

    # Wykres
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(tyre_ages, proba_uncal[:, 2], "o-", color="#d62728",
            linewidth=2.5, markersize=4, label="P(Stan 2) — przed kalibracją", alpha=0.8)
    ax.plot(tyre_ages, proba_cal[:, 2], "s-", color="#1f77b4",
            linewidth=2.5, markersize=4, label="P(Stan 2) — po kalibracji", alpha=0.8)

    # Zaznacz prawdę (gdzie rzeczywiście jest Stan 2)
    cliff_mask = y_true == 2
    if cliff_mask.any():
        ax.scatter(tyre_ages[cliff_mask], proba_uncal[cliff_mask, 2],
                   color="#d62728", s=150, marker="o", edgecolors="black",
                   linewidths=1, zorder=5, label="Stan 2 (rzeczywisty)")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Próg 0.5")
    ax.set_xlabel("Wiek opony [okrążenia]", fontsize=11)
    ax.set_ylabel("Prawdopodobieństwo", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"Live Feed Porównanie (Kalibracja) | {driver} | {gp_name} | "
        f"stint {stint_number} | {compound}",
        fontsize=12
    )

    plt.tight_layout()
    out_name = PROC / f"calibration_livefeed_comparison_{session_key}_{driver}_stint{stint_number}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Zapisano: {out_name}")
    plt.show()

# 5. GŁÓWNY PIPELINE

if __name__ == "__main__":
    print("Wczytywanie danych i modelu...")
    df = pl.read_parquet(PROC / "features_all.parquet")

    X, y, groups, feature_cols = prepare_data(df)
    splits = chronological_split(X, y, groups, n_splits=3)

    # Załaduj oba modele
    import pickle

    with open(PROC / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    try:
        with open(PROC / "xgb_calibrated_model.pkl", "rb") as f:
            xgb_calibrated = pickle.load(f)
        print("Załadowano model skalibrowany\n")
    except FileNotFoundError:
        print("Model skalibrowany nie znaleziony!")
        xgb_calibrated = None

    # Przygotuj dane testowe (z ostatniego folda)
    train_idx, test_idx = splits[-1]
    X_test, y_test = X[test_idx], y[test_idx]

    # 1. SHAP
    explainer, shap_values, X_sample, y_sample, _ = run_shap_analysis(
        xgb_model, X[test_idx], y[test_idx], feature_cols
    )

    # Beeswarm per klasa
    for i, name in enumerate(["Stan_0", "Stan_1", "Stan_2_cliff"]):
        plot_shap_beeswarm(shap_values, X_sample, feature_cols, i, name)

    plot_shap_bar_all_classes(shap_values, feature_cols)

    # 2. RAPORT KALIBRACJI
    if xgb_calibrated is not None:
        cal_report = compare_calibration(
            xgb_model, xgb_calibrated, X_test, y_test, feature_cols
        )

    # 3. EKSPERYMENT bez lap_time
    model_tel, f1_tel, tel_cols = experiment_no_lap_time(
        df, feature_cols, splits
    )
    print(f"\n📊 F1-macro pełny model:           0.6822")
    print(f"📊 F1-macro bez lap_time:          {f1_tel:.4f}")
    print(f"📊 Różnica (koszt usunięcia):       {0.6822 - f1_tel:+.4f}")

    # 4. LIVE FEED — stinty z clifem
    # Znajdź stinty gdzie wystąpił Stan 2
    cliff_stints = (
        df.filter(pl.col("target") == 2)
          .group_by(["session_key", "driver", "stint_number"])
          .agg(pl.len().alias("cliff_windows"))
          .sort("cliff_windows", descending=True)
          .head(5)
    )
    print("\nTop stinty z największą liczbą okien Stan 2:")
    print(cliff_stints)

    for i, row in enumerate(cliff_stints.iter_rows(named=True)):
        # Live feed z oryginalnym modelem
        live_feed_simulation(
            xgb_model, df, feature_cols,
            session_key=row["session_key"],
            driver=row["driver"],
            stint_number=row["stint_number"],
        )

        # Live feed z porównaniem (przed/po kalibracji)
        if xgb_calibrated is not None and i < 3:  # Pokaż dla 3 najlepszych
            live_feed_comparison(
                xgb_model, xgb_calibrated, df, feature_cols,
                session_key=row["session_key"],
                driver=row["driver"],
                stint_number=row["stint_number"],
            )