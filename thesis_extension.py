import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import pickle
from config import PROC, RANDOM_STATE
from models import prepare_data, chronological_split, sort_df

def run_additional_baselines(X, y, splits):
    print("\n" + "═" * 65)
    print("1. DODATKOWE MODELE: Logistic Regression (Baseline) & Sieć MLP")
    print("═" * 65)

    lr_f1s, mlp_f1s = [], []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Uzupełnienie ewentualnych NaN medianą (ważne dla LR i MLP)
        finite_tr = np.isfinite(X_tr)
        med = np.zeros(X_tr.shape[1], dtype=np.float32)
        for col in range(X_tr.shape[1]):
            values = X_tr[finite_tr[:, col], col]
            if values.size:
                med[col] = np.median(values)

        X_tr_clean = np.where(finite_tr, X_tr, med)
        X_te_clean = np.where(np.isfinite(X_te), X_te, med)

        # A. Wieloklasowa Regresja Logistyczna (zupełny baseline liniowy)
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=RANDOM_STATE
            ))
        ])
        lr_pipe.fit(X_tr_clean, y_tr)
        y_pred_lr = lr_pipe.predict(X_te_clean)
        f1_lr = f1_score(y_te, y_pred_lr, average="macro", zero_division=0)
        lr_f1s.append(f1_lr)

        # B. Sieć neuronowa MLP (Multilayer Perceptron)
        # Ważenie próbek dla MLP przez balansowanie lub architekturę 64-32
        mlp_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.01,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=50,
                early_stopping=True,
                random_state=RANDOM_STATE
            ))
        ])
        mlp_pipe.fit(
            X_tr_clean,
            y_tr,
            mlp__sample_weight = compute_sample_weight("balanced", y_tr),
        )
        y_pred_mlp = mlp_pipe.predict(X_te_clean)
        f1_mlp = f1_score(y_te, y_pred_mlp, average="macro", zero_division=0)
        mlp_f1s.append(f1_mlp)

        print(f"Fold {fold_i+1}: Logistic Regression F1 = {f1_lr:.4f} | MLP (64x32) F1 = {f1_mlp:.4f}")
        if fold_i == len(splits) - 1:
            print(f"\n--- Raport dla Fold {fold_i + 1}: Regresja Logistyczna ---")
            print(classification_report(
                y_te, y_pred_lr, labels=[0, 1, 2],
                target_names = ["Stan0", "Stan1", "Stan2"], zero_division = 0
            ))
            print(f"\n--- Raport dla Fold {fold_i + 1}: Sieć neuronowa MLP ---")
            print(classification_report(
                y_te, y_pred_mlp, labels=[0, 1, 2],
                target_names = ["Stan0", "Stan1", "Stan2"], zero_division = 0
            ))

    print(f"\nPODSUMOWANIE DO TABELI W PRACY INŻYNIERSKIEJ:")
    print(f"Logistic Regression: średnie F1-macro = {np.mean(lr_f1s):.3f} ± {np.std(lr_f1s):.3f} (Fold {fold_i + 1}: {lr_f1s[-1]:.3f})")
    print(f"MLP Neural Network:  średnie F1-macro = {np.mean(mlp_f1s):.3f} ± {np.std(mlp_f1s):.3f} (Fold {fold_i + 1}: {mlp_f1s[-1]:.3f})")
    return lr_f1s, mlp_f1s


def run_driver_segmentation(df: pl.DataFrame, model, X, y, splits):
    print("\n" + "═" * 65)
    print("2. SEGMENTACJA KIEROWCÓW (Styl jazdy i dokładność per kierowca)")
    print("═" * 65)

    df_sorted = sort_df(df)
    _, test_idx = splits[-1]
    X_te, y_te = X[test_idx], y[test_idx]
    df_test = df_sorted[test_idx]

    y_pred = model.predict(X_te)
    drivers = df_test["driver"].to_numpy()

    # 1. Tabela skuteczności modelu per kierowca w Foldzie 3
    driver_stats = []
    for drv in np.unique(drivers):
        mask = drivers == drv
        if mask.sum() < 50:
            continue
        acc = np.mean(y_pred[mask] == y_te[mask]) * 100
        f1_m = f1_score(y_te[mask], y_pred[mask], average="macro", zero_division=0)
        cliff_ratio = np.mean(y_te[mask] == 2) * 100
        driver_stats.append((drv, int(mask.sum()), acc, f1_m, cliff_ratio))

    driver_stats.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'Kierowca':<10} | {'Okna':>6} | {'Dokładność [%]':>14} | {'F1-macro':>10} | {'Udział Stanu 2 [%]':>18}")
    print("-" * 68)
    for drv, n_win, acc, f1_m, cr in driver_stats:
        print(f"{drv:<10} | {n_win:>6} | {acc:>13.1f}% | {f1_m:>10.3f} | {cr:>17.1f}%")

    # 2. Klasteryzacja stylów jazdy kierowców (K-Means na całym zbiorze)
    drv_profiles = (
        df_sorted.group_by("driver")
        .agg([
            pl.col("brake_max").mean().alias("agresja_hamowania"),
            pl.col("wot_pct").mean().alias("pelny_gaz_wot"),
            pl.col("speed_std").mean().alias("zmiennosc_predkosci"),
            pl.col("degradation_rate").std().alias("niestabilnosc_tempa"),
            pl.len().alias("n_okien")
        ])
        .filter(pl.col("n_okien") >= 500)
        .sort("driver")
    )

    feat_clust = drv_profiles.select(["agresja_hamowania", "pelny_gaz_wot", "zmiennosc_predkosci"]).to_numpy()
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_clust)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(feat_scaled)

    # Wykres segmentacji kierowców
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    cluster_Labels = ["Klaster 0 (Płynny / Zarządzanie)", "Klaster 1 (Zbalansowany)", "Klaster 2 (Agresywny / Wysoka energia)"]
    #cluster_labels = [f"Klaster {c}" for c in range(3)]

    x_val = drv_profiles["agresja_hamowania"].to_numpy()
    y_val = drv_profiles["pelny_gaz_wot"].to_numpy() * 100
    drv_names = drv_profiles["driver"].to_list()

    for c in range(3):
        m = clusters == c
        ax.scatter(x_val[m], y_val[m], c=colors[c], s=120, label=cluster_Labels[c], alpha=0.85, edgecolors="black")

    for i, txt in enumerate(drv_names):
        ax.annotate(txt, (x_val[i], y_val[i]), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlabel("Średnie maksymalne ciśnienie/sygnał hamowania w oknie (brake_max)")
    ax.set_ylabel("Udział jazdy z w pełni otwartą przepustnicą WOT [%]")
    ax.set_title("Segmentacja stylów jazdy kierowców F1 (sezon 2025) — K-Means")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PROC / "driver_segmentation.png", dpi=200, bbox_inches="tight")
    print(f"\nZapisano wykres segmentacji kierowców: {PROC / 'driver_segmentation.png'}")


def run_strict_ablation(df: pl.DataFrame, feature_cols: list, splits):
    print("\n" + "═" * 65)
    print("3. ŚCISŁY EKSPERYMENT ABLACYJNY (Czyste czujniki bez pochodnych czasu)")
    print("═" * 65)

    # Usuwamy WSZYSTKIE cechy powiązane z czasem okrążenia i tempem degradacji
    time_related = [c for c in feature_cols if ("lap_time" in c) or ("degradation_rate" in c)]
    pure_sensor_cols = [c for c in feature_cols if c not in time_related]

    print(f"Usunięto ({len(time_related)} cech): {time_related}")
    print(f"Pozostało ({len(pure_sensor_cols)} czystych cech z czujników): {pure_sensor_cols[:8]}...")

    df_sorted = sort_df(df)
    X_pure = df_sorted.select(pure_sensor_cols).to_numpy().astype(np.float32)
    y = df_sorted["target"].to_numpy().astype(np.int32)

    train_idx, test_idx = splits[-1]
    X_tr, y_tr = X_pure[train_idx], y[train_idx]
    X_te, y_te = X_pure[test_idx],  y[test_idx]

    class_counts = np.bincount(y_tr)
    weights = np.array([len(y_tr) / (3 * class_counts[c]) for c in y_tr])

    model_pure = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, tree_method="hist", verbosity=0
    )
    model_pure.fit(X_tr, y_tr, sample_weight=weights)
    y_pred = model_pure.predict(X_te)
    f1_pure = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\nF1-macro (wyłącznie surowe czujniki + tyre_life, bez degradation_rate): {f1_pure:.4f}")


if __name__ == "__main__":
    df = pl.read_parquet(PROC / "features_all.parquet")
    X, y, groups, feature_cols = prepare_data(df)
    splits = chronological_split(X, y, groups)

    run_additional_baselines(X, y, splits)

    with open(PROC / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    run_driver_segmentation(df, xgb_model, X, y, splits)
    run_strict_ablation(df, feature_cols, splits)