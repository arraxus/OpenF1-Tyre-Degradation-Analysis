import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings("ignore")
from config import PROC, METADATA_EXCLUDE, N_SPLITS, OPTUNA_TRIALS, RANDOM_STATE

# 1. PRZYGOTOWANIE DANYCH

def sort_df(df: pl.DataFrame) -> pl.DataFrame:
    # Jedno źródło dla sortowania chronologicznego
    sort_cols = (
        ["race_date", "driver_number", "stint_number", "tyre_life"]
        if "race_date" in df.columns
        else ["session_key", "driver_number", "stint_number", "tyre_life"]
    )
    return df.sort(sort_cols)

def prepare_data(df: pl.DataFrame) -> tuple:
    # Zwraca X (numpy), y (numpy), grupy GP (do TimeSeriesSplit), oraz listę nazw cech
    # Importowany zbiór metadanych/kolumn do wykluczenia
    exclude = set(METADATA_EXCLUDE)

    # Dodatkowe kolumny specyficzne dla przygotowania danych (kolumny współliniowe)
    exclude.update({
        "lap_time_rolling_mean",  # r ok. 1.0 z lap_time_norm_pct
        "race_date",
    })

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in (pl.Float64, pl.Float32,
                                        pl.Int64, pl.Int32, pl.Int8)]

    df = sort_df(df)

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.int32)

    # Grupy GP — jedna wartość per wyścig; race_date jeśli dostępna, inaczej session_key
    if "race_date" in df.columns:
        groups = df["race_date"].to_numpy()
    else:
        print("⚠️  Brak kolumny race_date — session_key do grup")
        print("   Kolejność foldów może nie odpowiadać kalendarzowi")
        groups = df["session_key"].to_numpy()

    print(f"Cechy: {len(feature_cols)}")
    print(f"Próbki: {X.shape[0]}")
    print(f"Rozkład: {np.bincount(y)}")
    print(f"Wyścigi (session_key): {np.unique(groups)}")

    return X, y, groups, feature_cols


def chronological_split(X, y, groups, n_splits=N_SPLITS):
    # TimeSeriesSplit na poziomie wyścigów. Trenujemy na wcześniejszych GP, testujemy na późniejszych.
    unique_sessions = np.unique(groups)
    n = len(unique_sessions)

    splits = []
    # Podział: pierwsze k GP jako trening, następne jako test
    step = max(1, n // (n_splits + 1))
    for i in range(1, n_splits + 1):
        train_sessions = unique_sessions[:i * step]
        test_sessions  = unique_sessions[i * step: (i + 1) * step]
        if len(test_sessions) == 0:
            continue
        train_idx = np.where(np.isin(groups, train_sessions))[0]
        test_idx  = np.where(np.isin(groups, test_sessions))[0]
        splits.append((train_idx, test_idx))
        print(f"  Fold {i}: train={len(train_sessions)} GP "
              f"({len(train_idx)} wierszy), "
              f"test={len(test_sessions)} GP "
              f"({len(test_idx)} wierszy)")
        print(f"    Trening:  {train_sessions}")
        print(f"    Test:     {test_sessions}")

    return splits


# 2. DECISION TREE — baseline interpretowalny

def train_decision_tree(X, y, splits, feature_cols):
    # Trenuje Decision Tree na każdym foldzie, wybiera najlepszy
    print("\n" + "═" * 60)
    print("MODEL 1 — Decision Tree (baseline)")
    print("═" * 60)

    # Wagi klas — kompensacja niezbalansowania
    class_counts = np.bincount(y)
    class_weight  = {
        i: len(y) / (len(class_counts) * c)
        for i, c in enumerate(class_counts)
    }
    print(f"Wagi klas: {class_weight}")

    results = []
    best_model = None
    best_f1    = -1
    last_y_te = None
    last_y_pred = None

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        dt = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        dt.fit(X_tr, y_tr)
        y_pred = dt.predict(X_te)

        # Zachowaj predykcje dla ostatniego foldu (do confusion matrix)
        if fold_i == len(splits) - 1:
            last_y_te = y_te
            last_y_pred = y_pred

        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        results.append(f1)
        print(f"\nFold {fold_i + 1}: F1-macro = {f1:.4f}")
        print(classification_report(y_te, y_pred,
                                    target_names=["Stan0","Stan1","Stan2"],
                                    zero_division=0))

        if f1 > best_f1:
            best_f1    = f1
            best_model = dt

    print(f"\nŚrednie F1-macro: {np.mean(results):.4f} ± {np.std(results):.4f}")

    # Eksport reguł tekstowych (top 3 poziomy)
    rules = export_text(best_model, feature_names=feature_cols, max_depth=3)
    print("\nReguły decyzyjne (głębokość ≤ 3):")
    print(rules)
    (PROC / "dt_rules.txt").write_text(rules)

    # Wykres drzewa
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        best_model, feature_names=feature_cols,
        class_names=["Stan0","Stan1","Stan2"],
        filled=True, max_depth=3, ax=ax, fontsize=7,
        impurity=False, proportion=True
    )
    ax.set_title("Decision Tree — top 3 poziomy", fontsize=13)
    plt.tight_layout()
    plt.savefig(PROC / "dt_tree.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Confusion matrix (ostatni fold) — użyj predykcji z pętli
    if last_y_te is not None and last_y_pred is not None:
        plot_confusion_matrix(last_y_te, last_y_pred, "Decision Tree", "dt_cm.png")

    return best_model, np.mean(results)


# 3. XGBOOST — model główny

def objective_xgb(trial, X, y, splits):
    # Funkcja celu dla Optuna
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "random_state":     RANDOM_STATE,
        "tree_method":      "hist",
        "device":           "cpu",
    }

    f1s = []
    for train_idx, test_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Wagi próbek — kompensacja niezbalansowania
        class_counts = np.bincount(y_tr)
        weights = np.array([
            len(y_tr) / (3 * class_counts[c]) for c in y_tr
        ])

        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(X_tr, y_tr, sample_weight=weights,
                  eval_set=[(X_te, y_te)], verbose=False)
        y_pred = model.predict(X_te)
        f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
    return np.mean(f1s)


def train_xgboost(X, y, splits, feature_cols):
    # Trenuje XGBoost z optymalizacją Optuna
    print("\n" + "═" * 60)
    print("MODEL 2 — XGBoost + Optuna")
    print("═" * 60)

    # Optymalizacja hiperparametrów
    print(f"\nOptuna: {OPTUNA_TRIALS} prób...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective_xgb(trial, X, y, splits),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update({
        "objective":    "multi:softprob",
        "num_class":    3,
        "eval_metric":  "mlogloss",
        "random_state": RANDOM_STATE,
        "tree_method":  "hist",
        "device":       "cpu",
        "verbosity":    0,
    })
    print(f"\nNajlepsze parametry: {best_params}")
    print(f"Najlepsze F1-macro (CV): {study.best_value:.4f}")

    # Trening finalny na wszystkich danych oprócz ostatniego foldu
    train_idx, test_idx = splits[-1]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    class_counts = np.bincount(y_tr)
    weights = np.array([len(y_tr) / (3 * class_counts[c]) for c in y_tr])

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_tr, y_tr,
        sample_weight=weights,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    # Predykcje przed kalibracją
    y_pred = final_model.predict(X_te)

    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\nF1-macro (test fold): {f1:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Stan0", "Stan1", "Stan2"],
                                zero_division=0))

    # Utwórz bazowy estimator (nowy, nie prefit)
    base_for_cal = xgb.XGBClassifier(**best_params)

    # Kalibracja prawdopodobieństw na osobnym zbiorze walidacyjnym
    # cv=3 (albo inna liczba >=2) method='sigmoid' dla małej liczby próbek/klas rzadkich
    calibrated_model = CalibratedClassifierCV(
        estimator=base_for_cal,
        method="sigmoid",
        cv=3
    )
    # Fit kalibratora na zbiorze walidacyjnym (X_te, y_te)
    calibrated_model.fit(X_te, y_te)

    # Probki i predykcje po kalibracji
    y_proba_before = final_model.predict_proba(X_te)
    y_proba_after = calibrated_model.predict_proba(X_te)
    y_pred_cal = np.argmax(y_proba_after, axis=1)

    # Ocena
    f1_cal = f1_score(y_te, y_pred_cal, average="macro", zero_division=0)
    print(f"\nF1-macro po kalibracji (na zbiorze walidacyjnym): {f1_cal:.4f}")
    print(classification_report(y_te, y_pred_cal,
                                target_names=["Stan0", "Stan1", "Stan2"],
                                zero_division=0))

    # Reliability diagram dla klasy 2
    plot_calibration_curve(
        y_te,
        y_proba_before,
        proba_after=y_proba_after,
        class_idx=2,
        fname="calibration_curve_stan2.png"
    )

    # Confusion matrix i feature importance
    plot_confusion_matrix(y_te, y_pred, "XGBoost", "xgb_cm.png")
    plot_feature_importance(final_model, feature_cols)

    # Zapis modelu
    import pickle
    with open(PROC / "xgb_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(PROC / "xgb_calibrated_model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    print(f"Model zapisany: {PROC / 'xgb_model.pkl'}")
    print(f"Model skalibrowany zapisany: {PROC / 'xgb_calibrated_model.pkl'}")

    return final_model, f1, best_params


# 4. NARZĘDZIA POMOCNICZE

def plot_confusion_matrix(y_true, y_pred, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stan 0", "Stan 1", "Stan 2"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(PROC / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_cols: list, top_n: int = 20):
    # Top-N najważniejszych cech wg XGBoost (gain)
    importance = model.get_booster().get_score(importance_type="gain")
    # Mapuj indeksy f0,f1,... na nazwy cech
    named = {}
    for k, v in importance.items():
        try:
            idx = int(k[1:])
            named[feature_cols[idx]] = v
        except (ValueError, IndexError):
            named[k] = v

    sorted_imp = sorted(named.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), vals, color="#2E75B6")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gain (ważność cechy)")
    ax.set_title(f"Top {top_n} cech — XGBoost")
    plt.tight_layout()
    plt.savefig(PROC / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_curve(y_true, proba_before, proba_after=None, class_idx: int = 2, fname: str = "calibration_curve.png"):
    """
    Reliability diagram dla wskazanej klasy.
    Domyślnie class_idx=2, bo to najbardziej interesująca i rzadka klasa.
    """
    y_true_bin = (np.asarray(y_true) == class_idx).astype(int)

    frac_pos_b, mean_pred_b = calibration_curve(
        y_true_bin, proba_before[:, class_idx], n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred_b, frac_pos_b, "o-", label="XGBoost przed kalibracją", color="#d62728")

    if proba_after is not None:
        frac_pos_a, mean_pred_a = calibration_curve(
            y_true_bin, proba_after[:, class_idx], n_bins=10, strategy="uniform"
        )
        ax.plot(mean_pred_a, frac_pos_a, "o-", label="Po kalibracji", color="#1f77b4")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Idealna kalibracja")
    ax.set_xlabel("Średnie przewidywane prawdopodobieństwo")
    ax.set_ylabel("Rzeczywisty odsetek pozytywnych")
    ax.set_title(f"Calibration curve — klasa Stan {class_idx}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROC / fname, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(results: dict):
    # Tabela porównawcza wszystkich modeli
    print("\n" + "═" * 60)
    print("PORÓWNANIE MODELI")
    print("═" * 60)
    print(f"{'Model':<20} {'F1-macro':>10}")
    print("-" * 32)
    for name, f1 in results.items():
        print(f"{name:<20} {f1:>10.4f}")


# 5. GŁÓWNY PIPELINE

if __name__ == "__main__":
    print("Wczytywanie danych...")
    df = pl.read_parquet(PROC / "features_all.parquet")
    print(f"Shape: {df.shape}")

    # Przygotowanie
    X, y, groups, feature_cols = prepare_data(df)
    splits = chronological_split(X, y, groups)

    model_results = {}

    # Model 1 — Decision Tree
    dt_model, dt_f1 = train_decision_tree(X, y, splits, feature_cols)
    model_results["Decision Tree"] = dt_f1

    # Model 2 — XGBoost
    xgb_model, xgb_f1, best_params = train_xgboost(X, y, splits, feature_cols)
    model_results["XGBoost"] = xgb_f1

    compare_models(model_results)