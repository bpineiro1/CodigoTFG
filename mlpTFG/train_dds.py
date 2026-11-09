"""
MLP + DDS (selección dinámica de características) evaluado con el MISMO protocolo que
el MLP baseline (train_mlp.py):
  - Las mismas semillas, folds externos/internos, número de candidatos y métrica interna.
  - Los hiperparámetros comunes (épocas, batch, lr, pesos, weight decay, dropout) se
    sortean igual que en el baseline: el candidato i del MLP+DDS tiene los mismos valores
    que el candidato i del MLP, más K y el peso de la penalización.
  - El umbral de decisión se elige igual (G-Mean en el bucle interno, trasladado al test
    como porcentaje de positivos).
La configuración común se importa de train_mlp.py para que no pueda divergir.

Además del rendimiento, guarda cuántas características usa cada gen y con qué
frecuencia se selecciona cada característica en genes CR y NotCR.

Uso (desde la raíz del proyecto, junto a train_mlp.py):
    python train_dds.py                      # 10 repeticiones, GO y PathDip
    python train_dds.py --repeats 2 --datasets GO
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from Code.model import MLP_DDS
from Code.metrics import calculate_all_metrics, best_threshold, predict_top_fraction
from Code.plots import generate_evaluation_plots, generate_summary_table_image

# Configuración común con el baseline
from train_mlp import (
    SEEDS, N_ITER_SEARCH, N_OUTER_SPLITS, N_INNER_SPLITS, INNER_METRIC, THRESHOLD_CRITERION,
    PARAM_SPACE, REPORT_METRICS, RESULTS_DIR, REPS_DIR, set_seed, fmt,
)

if hasattr(os, "cpu_count") and os.cpu_count() is not None:
    torch.set_num_threads(os.cpu_count())


# =====================================================================
# HIPERPARÁMETROS PROPIOS DEL DDS
# =====================================================================
# K (máximo de características activas que usa cada gen). Rejilla adaptada a cada dataset
# según el nº de características activas por gen:
#   GO:      mediana 94 (percentiles 25-75: 44-170)
#   PathDip: mediana 11 (percentiles 25-75: 4-26)
# Los valores más altos equivalen casi a no seleccionar, así que la búsqueda puede
# "decidir" que no conviene podar.
K_GRID = {
    "GO": [10, 25, 50, 100, 200, 400],
    "PathDip": [3, 5, 10, 20, 50, 100],
}
PENALTY_GRID = [0.0, 1e-3, 1e-2, 1e-1]

MODEL_TAG = "MLP_DDS"


def load_data(dataset_name: str):
    path = f"./Data/{dataset_name}.csv"
    raw_df = pd.read_csv(path)

    gene_col = raw_df.columns[0]
    gene_names = raw_df[gene_col].values

    if "Class" in raw_df.columns:
        y = (raw_df["Class"].astype(str).str.strip() == "CR").values.astype(int)
    elif "DR" in raw_df.columns:
        y = raw_df["DR"].values.astype(int)
    else:
        y = (raw_df.iloc[:, -1].astype(str).str.strip() == "CR").values.astype(int)

    cols_to_drop = [gene_col, "Class", "DR", "Unnamed: 0", "Symbol", "Entrez", "id", "gene"]
    feature_df = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])
    numeric_df = feature_df.select_dtypes(include=[np.number])
    x = numeric_df.values.astype(np.float32)
    feature_names = numeric_df.columns.values

    print(f"[{dataset_name}] Instancias: {x.shape[0]} | Características: {x.shape[1]} | Positivos (CR): {np.sum(y)} / {len(y)}")
    return x, y, gene_names, feature_names


def sample_candidates(seed, outer_fold, dataset):
    """
    Los hiperparámetros comunes se sortean con el mismo generador y en el mismo orden que
    en train_mlp.py; los propios del DDS, con un generador aparte para no alterar esa secuencia.
    """
    rng_common = random.Random(seed + outer_fold)
    rng_dds = random.Random(seed + outer_fold + 1_000_003)
    candidates = []
    for _ in range(N_ITER_SEARCH):
        cand = {k: rng_common.choice(v) for k, v in PARAM_SPACE.items()}
        cand["k_features"] = rng_dds.choice(K_GRID[dataset])
        cand["penalty_weight"] = rng_dds.choice(PENALTY_GRID)
        candidates.append(cand)
    return candidates


def train_single_dds_model(x_tr, y_tr, params, seed, device="cpu"):
    """Entrena un MLP+DDS con una combinación fija de hiperparámetros."""
    set_seed(seed)

    model = MLP_DDS(
        input_dim=x_tr.shape[1],
        n_features_to_select=params["k_features"],
        hidden_dim1=50,
        hidden_dim2=25,
        dropout_rate=params.get("dropout", 0.2),
    ).to(device)

    n_pos = np.sum(y_tr)
    n_neg = len(y_tr) - n_pos
    if params["weight_mode"] == "moderate":
        pos_weight_val = np.sqrt(n_neg / max(n_pos, 1))
    elif params["weight_mode"] == "full":
        pos_weight_val = n_neg / max(n_pos, 1)
    else:
        pos_weight_val = 1.0

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))

    dataset = TensorDataset(
        torch.tensor(x_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, generator=generator)

    model.train()
    for _ in range(params["epochs"]):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, penalty, _, _ = model(bx)
            loss = criterion(logits, by) + params["penalty_weight"] * penalty
            loss.backward()
            optimizer.step()

    return model


def evaluate_dds_model(model, x_eval, device="cpu"):
    """Devuelve probabilidades, máscaras de selección (solo variables activas) y nº de variables usadas."""
    model.eval()
    with torch.no_grad():
        logits, _, hard_mask, n_used = model(torch.tensor(x_eval, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        masks = hard_mask.cpu().numpy()
        n_used = n_used.cpu().numpy()
    return probs, masks, n_used


def nested_cv_dds(x, y, dataset, seed, device="cpu"):
    """Una repetición de la validación cruzada anidada con la semilla seed."""
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed % (2**32))
    folds_records = []
    oof_probs = np.full(len(y), np.nan)

    # Frecuencia de selección por característica y clase (sobre los genes de test)
    n_feat = x.shape[1]
    sel_counts = {1: np.zeros(n_feat), 0: np.zeros(n_feat)}
    act_counts = {1: np.zeros(n_feat), 0: np.zeros(n_feat)}

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), 1):
        x_outer_train, x_outer_test = x[train_idx], x[test_idx]
        y_outer_train, y_outer_test = y[train_idx], y[test_idx]

        # Bucle interno: Random Search
        inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=(seed + outer_fold) % (2**32))
        candidates = sample_candidates(seed, outer_fold, dataset)

        best_inner_score, best_candidate, best_inner_oof = -1.0, None, None
        print(f"    Fold {outer_fold:02d}/{N_OUTER_SPLITS} ", end="", flush=True)

        for cand_idx, cand in enumerate(candidates):
            inner_scores = []
            inner_oof = np.zeros(len(y_outer_train))
            for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_cv.split(x_outer_train, y_outer_train)):
                train_seed = seed + outer_fold * 1000 + cand_idx * 10 + in_fold
                inner_model = train_single_dds_model(
                    x_outer_train[in_tr_idx], y_outer_train[in_tr_idx], cand, seed=train_seed, device=device
                )
                val_probs, _, _ = evaluate_dds_model(inner_model, x_outer_train[in_val_idx], device=device)
                inner_oof[in_val_idx] = val_probs
                inner_scores.append(calculate_all_metrics(y_outer_train[in_val_idx], val_probs)[INNER_METRIC])
            print(".", end="", flush=True)

            mean_score = float(np.mean(inner_scores))
            if mean_score > best_inner_score:
                best_inner_score, best_candidate, best_inner_oof = mean_score, cand, inner_oof

        # Umbral: igual que en el baseline (G-Mean en el bucle interno, trasladado como porcentaje)
        threshold = best_threshold(y_outer_train, best_inner_oof, criterion=THRESHOLD_CRITERION)
        positive_fraction = float(np.mean(best_inner_oof >= threshold))

        # Reentrenamiento final y test
        final_seed = seed + outer_fold * 1000 + 999
        final_model = train_single_dds_model(x_outer_train, y_outer_train, best_candidate, seed=final_seed, device=device)
        test_probs, test_masks, test_n_used = evaluate_dds_model(final_model, x_outer_test, device=device)
        oof_probs[test_idx] = test_probs

        test_pred = predict_top_fraction(test_probs, positive_fraction)
        test_m = calculate_all_metrics(y_outer_test, test_probs, y_pred=test_pred)

        n_active = (x_outer_test > 0).sum(axis=1)
        for cls in (0, 1):
            idx = y_outer_test == cls
            sel_counts[cls] += test_masks[idx].sum(axis=0)
            act_counts[cls] += (x_outer_test[idx] > 0).sum(axis=0)

        record = {
            "Semilla": seed,
            "Fold": outer_fold,
            "K": best_candidate["k_features"],
            "Penalty": best_candidate["penalty_weight"],
            "Epochs": best_candidate["epochs"],
            "Batch": best_candidate["batch_size"],
            "LR": best_candidate["lr"],
            "Weight": best_candidate["weight_mode"],
            "WeightDecay": best_candidate["weight_decay"],
            "Dropout": best_candidate["dropout"],
            f"Inner_{INNER_METRIC}": best_inner_score,
            "Umbral_interno": threshold,
            "Pct_positivos": positive_fraction,
            "Feats_activas_media": float(n_active.mean()),
            "Feats_usadas_media": float(test_n_used.mean()),
            "Pct_activas_usadas": float(100 * test_n_used.sum() / max(n_active.sum(), 1)),
        }
        record.update({name: test_m[key] for key, name in REPORT_METRICS.items()})
        record["Precision"] = test_m["precision"]
        folds_records.append(record)

        print(f" K={best_candidate['k_features']} | usa {test_n_used.mean():.0f} de {n_active.mean():.0f} activas | "
              f"AUC-PR={test_m['auc_pr']:.3f} | Sens={test_m['sensitivity']:.3f} | Espec={test_m['specificity']:.3f}")

    df_freq = pd.DataFrame({
        "Sel_CR": sel_counts[1], "Act_CR": act_counts[1],
        "Sel_NotCR": sel_counts[0], "Act_NotCR": act_counts[0],
    })
    return folds_records, oof_probs, df_freq


def run_or_load_repetition(x, y, dataset, seed, device):
    """Ejecuta una repetición o la carga de disco si ya se había terminado antes."""
    paths = {
        "folds": os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_folds.csv"),
        "oof": os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_oof.csv"),
        "freq": os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_freq.csv"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  [Semilla {seed}] ya calculada, se carga de '{REPS_DIR}/'")
        return pd.read_csv(paths["folds"]), pd.read_csv(paths["oof"])["prob"].values, pd.read_csv(paths["freq"])

    folds_records, oof, df_freq = nested_cv_dds(x, y, dataset, seed, device=device)
    df_folds = pd.DataFrame(folds_records)
    os.makedirs(REPS_DIR, exist_ok=True)
    df_folds.to_csv(paths["folds"], index=False)
    pd.DataFrame({"prob": oof}).to_csv(paths["oof"], index=False)
    df_freq.to_csv(paths["freq"], index=False)
    return df_folds, oof, df_freq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["GO", "PathDip"])
    parser.add_argument("--repeats", type=int, default=len(SEEDS), help="nº de repeticiones (semillas) a usar")
    args = parser.parse_args()

    seeds = SEEDS[: args.repeats]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== MLP + DDS: VALIDACIÓN CRUZADA ANIDADA REPETIDA ===")
    print(f"Dispositivo: {device} | Repeticiones: {len(seeds)} | Folds: {N_OUTER_SPLITS}x{N_INNER_SPLITS} | "
          f"Candidatos por fold: {N_ITER_SEARCH} | Métrica interna: {INNER_METRIC} | Umbral por: {THRESHOLD_CRITERION}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows, memoria_rows = [], []

    for dataset in args.datasets:
        if not os.path.exists(f"./Data/{dataset}.csv"):
            print(f"[AVISO] No se encontró ./Data/{dataset}.csv, omitiendo...")
            continue
        if dataset not in K_GRID:
            print(f"[AVISO] No hay rejilla de K definida para {dataset}, omitiendo...")
            continue

        x, y, gene_names, feature_names = load_data(dataset)
        all_folds, rep_rows, oof_matrix = [], [], []
        freq_total = None

        for r, seed in enumerate(seeds, 1):
            print(f"\n  ===== {dataset} | Repetición {r}/{len(seeds)} (semilla {seed}) =====")
            df_folds, oof, df_freq = run_or_load_repetition(x, y, dataset, seed, device)
            all_folds.append(df_folds.assign(Repeticion=r))
            oof_matrix.append(oof)
            freq_total = df_freq if freq_total is None else freq_total + df_freq

            rep_row = {"Repeticion": r, "Semilla": seed}
            rep_row.update({name: df_folds[name].mean() for name in REPORT_METRICS.values()})
            rep_row["Feats_usadas_media"] = df_folds["Feats_usadas_media"].mean()
            rep_row["Pct_activas_usadas"] = df_folds["Pct_activas_usadas"].mean()
            rep_rows.append(rep_row)
            print(f"  -> Repetición {r}: AUC-ROC={rep_row['AUC-ROC']:.3f} | AUC-PR={rep_row['AUC-PR']:.3f} | "
                  f"G-Mean={rep_row['G-Mean']:.3f} | usa el {rep_row['Pct_activas_usadas']:.0f}% de las activas")

        # Ficheros por dataset
        df_all_folds = pd.concat(all_folds, ignore_index=True)
        df_all_folds.to_csv(os.path.join(RESULTS_DIR, f"metricas_detalladas_folds_{MODEL_TAG}_{dataset}.csv"), index=False)

        df_reps = pd.DataFrame(rep_rows)
        df_reps.to_csv(os.path.join(RESULTS_DIR, f"metricas_por_repeticion_{MODEL_TAG}_{dataset}.csv"), index=False)

        oof_matrix = np.vstack(oof_matrix)
        df_rank = pd.DataFrame({"Gen": gene_names, "Clase_CR": y, "Prob_media": oof_matrix.mean(axis=0)})
        for r, seed in enumerate(seeds, 1):
            df_rank[f"Prob_seed{seed}"] = oof_matrix[r - 1]
        df_rank.sort_values("Prob_media", ascending=False).to_csv(
            os.path.join(RESULTS_DIR, f"ranking_genes_{MODEL_TAG}_{dataset}.csv"), index=False)

        # Frecuencia de selección de cada característica (sobre todas las repeticiones):
        #   Sel_%: en qué % de los genes de la clase se seleccionó
        #   Sel_si_activa_%: de los genes de la clase que la tienen activa, en qué % se seleccionó
        df_feat = pd.DataFrame({"Caracteristica": feature_names})
        n_cls = {"CR": len(seeds) * int(np.sum(y == 1)), "NotCR": len(seeds) * int(np.sum(y == 0))}
        for cls in ("CR", "NotCR"):
            df_feat[f"Sel_{cls}_%"] = 100 * freq_total[f"Sel_{cls}"].values / n_cls[cls]
            df_feat[f"Activa_{cls}_%"] = 100 * freq_total[f"Act_{cls}"].values / n_cls[cls]
            df_feat[f"Sel_si_activa_{cls}_%"] = 100 * freq_total[f"Sel_{cls}"].values / np.maximum(freq_total[f"Act_{cls}"].values, 1)
        df_feat["Diferencia_CR_menos_NotCR"] = df_feat["Sel_CR_%"] - df_feat["Sel_NotCR_%"]
        df_feat = df_feat.sort_values("Sel_CR_%", ascending=False)
        df_feat.round(2).to_csv(os.path.join(RESULTS_DIR, f"frecuencia_seleccion_{MODEL_TAG}_{dataset}.csv"), index=False)

        print(f"\n---> Características más seleccionadas en genes CR ({dataset}):")
        print(df_feat.head(10)[["Caracteristica", "Sel_CR_%", "Sel_NotCR_%", "Sel_si_activa_CR_%"]].round(1).to_string(index=False))

        runs_metrics = [
            [{key: row[name] for key, name in REPORT_METRICS.items()} for _, row in df.iterrows()]
            for df in all_folds
        ]
        generate_evaluation_plots(runs_metrics, output_dir=os.path.join(RESULTS_DIR, f"plots_{dataset}"),
                                  dataset_name=dataset, model_name="MLP_DDS")

        # Media ± desviación típica entre repeticiones
        cols = list(REPORT_METRICS.values())
        means = df_reps[cols].mean()
        stds = df_reps[cols].std(ddof=1) if len(df_reps) > 1 else means * 0
        feats_txt = (f"{df_reps['Feats_usadas_media'].mean():.0f} por gen "
                     f"({df_reps['Pct_activas_usadas'].mean():.0f}% de las activas)")

        summary_rows.append({
            "Fase": "MLP + DDS",
            "Modelo": f"MLP + DDS ({dataset})",
            "Nº Feats": feats_txt,
            "G-Mean": fmt(means["G-Mean"], stds["G-Mean"]),
            "F1-Score": fmt(means["F1-Score"], stds["F1-Score"]),
            "AUC-ROC": fmt(means["AUC-ROC"], stds["AUC-ROC"]),
            "AUC-PR": fmt(means["AUC-PR"], stds["AUC-PR"]),
        })
        memoria_row = {"Método": "MLP + DDS", "Dataset": dataset, "Nº Feats": feats_txt}
        memoria_row.update({name: fmt(means[name], stds[name]) for name in cols})
        memoria_rows.append(memoria_row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, f"tabla_resumen_{MODEL_TAG}_NestedCV.csv"), index=False)
        generate_summary_table_image(summary_rows, os.path.join(RESULTS_DIR, f"tabla_resumen_{MODEL_TAG}_NestedCV.png"))

        df_memoria = pd.DataFrame(memoria_rows)
        df_memoria.to_csv(os.path.join(RESULTS_DIR, "tabla_dds_media_std.csv"), index=False)

        print("\n==========================================================================")
        print(f"RESULTADOS: media ± desviación típica entre {len(seeds)} repeticiones")
        print("==========================================================================")
        print(df_memoria.to_string(index=False))
        print(f"\n[OK] Ficheros guardados en '{RESULTS_DIR}/'.")
