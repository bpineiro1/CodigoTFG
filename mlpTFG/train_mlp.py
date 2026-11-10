"""
MLP baseline (sin selección de características) evaluado con validación cruzada
anidada repetida.

Protocolo (el mismo que debe usar el script del MLP+DDS):
  - Para cada semilla de SEEDS se ejecuta una validación cruzada anidada completa
    (N_OUTER_SPLITS folds externos x N_INNER_SPLITS folds internos con Random Search).
  - En cada repetición se calcula la media de las métricas sobre los folds externos.
  - El resultado final se expresa como media ± desviación típica (ddof=1) entre las
    repeticiones.

Las repeticiones ya terminadas se guardan en <carpeta de resultados>/reps/ y no se
recalculan si el script se vuelve a lanzar. Además de las probabilidades de test, se
guardan las predicciones de validación interna del candidato elegido en cada fold, para
poder recalcular el umbral con otro criterio sin volver a entrenar.

Umbral de decisión (--umbral): se elige en el bucle interno maximizando G-Mean (por
defecto, resultados en results/) o F1 (resultados en results/umbral_f1/). El criterio
del umbral no cambia los modelos entrenados ni los hiperparámetros elegidos (eso lo
decide INNER_METRIC), así que las métricas que no dependen del umbral (AUC-ROC, AUC-PR,
NDCG@10, Precision@10) salen idénticas con los dos criterios.

Uso (desde la raíz del proyecto, donde están Code/ y Data/):
    python train_mlp.py                      # 10 repeticiones, GO y PathDip
    python train_mlp.py --repeats 5          # solo las 5 primeras semillas
    python train_mlp.py --datasets PathDip
    python train_mlp.py --umbral f1          # umbral elegido por F1
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

from Code.model import SimpleMLP
from Code.metrics import calculate_all_metrics, best_threshold, predict_top_fraction
from Code.plots import generate_evaluation_plots, generate_summary_table_image

if hasattr(os, "cpu_count") and os.cpu_count() is not None:
    torch.set_num_threads(os.cpu_count())


# =====================================================================
# CONFIGURACIÓN GLOBAL (debe coincidir con la del script del MLP+DDS)
# =====================================================================
# Semillas de las repeticiones: las mismas que en el TFG de referencia
SEEDS = [14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903]
SEED = 42                # semilla usada por check_overfitting.py (no afecta a este script)

N_ITER_SEARCH = 20       # candidatos de Random Search por partición externa
N_OUTER_SPLITS = 10      # particiones del bucle externo
N_INNER_SPLITS = 3       # particiones del bucle interno
INNER_METRIC = "auc_pr"  # métrica del bucle interno (elige hiperparámetros, no depende del umbral)
THRESHOLD_CRITERION = "gmean"  # el umbral de decisión se elige en el bucle interno maximizando esta métrica
                               # ("gmean" o "f1"; se cambia con --umbral)

# El ORDEN de las claves importa: el MLP+DDS debe sortear estas mismas claves
# en este mismo orden con la misma semilla para compartir hiperparámetros.
PARAM_SPACE = {
    "epochs": [5, 10, 15, 20, 30, 40],   # el sobreajuste empieza en las épocas 4-17
    "batch_size": [32, 64],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_mode": ["moderate", "full"],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "dropout": [0.2, 0.5],
}

# Métricas
REPORT_METRICS = {
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "gmean": "G-Mean",
    "f1": "F1-Score",
    "ndcg_at_10": "NDCG@10",
    "precision_at_10": "Precision@10",
    "recall_at_10": "Recall@10",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}
MODEL_TAG = "MLP"
RESULTS_DIR = "results"
REPS_DIR = os.path.join(RESULTS_DIR, "reps")


def results_dir_for(criterion: str) -> str:
    """Carpeta de resultados según el criterio del umbral (G-Mean conserva la de la versión 1)."""
    return "results" if criterion == "gmean" else os.path.join("results", f"umbral_{criterion}")


def set_seed(seed: int):
    """Fija todas las fuentes de aleatoriedad para poder reproducir el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    print(f"[{dataset_name}] Instancias: {x.shape[0]} | Características: {x.shape[1]} | Positivos (CR): {np.sum(y)} / {len(y)}")
    return x, y, gene_names


def train_single_mlp_model(x_tr, y_tr, params, seed, device="cpu"):
    """Entrena una red SimpleMLP con una combinación fija de hiperparámetros."""
    set_seed(seed)

    model = SimpleMLP(
        input_dim=x_tr.shape[1],
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
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    return model


def evaluate_mlp_model(model, x_eval, device="cpu"):
    """Evalúa el modelo devolviendo probabilidades continuas."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_eval, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


def nested_cv_mlp_random_search(x, y, seed, device="cpu"):
    """
    Una repetición de la validación cruzada anidada con la semilla `seed`:
      - Bucle externo: estimación del rendimiento sin sesgo de optimismo.
      - Bucle interno con Random Search: elección de hiperparámetros según INNER_METRIC.
    Devuelve (registros por fold, métricas por fold, probabilidades fuera de fold).
    """
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed % (2**32))
    folds_records, fold_metrics = [], []
    oof_probs = np.full(len(y), np.nan)
    oof_fold = np.zeros(len(y), dtype=int)
    inner_rows = []  # predicciones de validación interna del candidato elegido (para recalcular umbrales)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), 1):
        x_outer_train, x_outer_test = x[train_idx], x[test_idx]
        y_outer_train, y_outer_test = y[train_idx], y[test_idx]

        # Bucle interno: Random Search
        inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=(seed + outer_fold) % (2**32))
        rng = random.Random(seed + outer_fold)
        candidates = [{k: rng.choice(v) for k, v in PARAM_SPACE.items()} for _ in range(N_ITER_SEARCH)]

        best_inner_score, best_candidate, best_inner_oof = -1.0, None, None
        print(f"    Fold {outer_fold:02d}/{N_OUTER_SPLITS} ", end="", flush=True)

        for cand_idx, cand in enumerate(candidates):
            inner_scores = []
            inner_oof = np.zeros(len(y_outer_train))
            for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_cv.split(x_outer_train, y_outer_train)):
                train_seed = seed + outer_fold * 1000 + cand_idx * 10 + in_fold
                inner_model = train_single_mlp_model(
                    x_outer_train[in_tr_idx], y_outer_train[in_tr_idx], cand, seed=train_seed, device=device
                )
                val_probs = evaluate_mlp_model(inner_model, x_outer_train[in_val_idx], device=device)
                inner_oof[in_val_idx] = val_probs
                inner_scores.append(calculate_all_metrics(y_outer_train[in_val_idx], val_probs)[INNER_METRIC])
            print(".", end="", flush=True)

            mean_score = float(np.mean(inner_scores))
            if mean_score > best_inner_score:
                best_inner_score, best_candidate, best_inner_oof = mean_score, cand, inner_oof

        # Umbral de decisión: se elige con las predicciones de validación interna del mejor
        # candidato (el test externo no interviene). Como el modelo final no tiene la misma
        # escala de probabilidades que los modelos internos, el umbral se traslada como
        # porcentaje de genes marcados como positivos, no como valor de probabilidad.
        threshold = best_threshold(y_outer_train, best_inner_oof, criterion=THRESHOLD_CRITERION)
        positive_fraction = float(np.mean(best_inner_oof >= threshold))

        # Reentrenamiento final y test
        final_seed = seed + outer_fold * 1000 + 999
        final_model = train_single_mlp_model(x_outer_train, y_outer_train, best_candidate, seed=final_seed, device=device)
        test_probs = evaluate_mlp_model(final_model, x_outer_test, device=device)
        oof_probs[test_idx] = test_probs
        oof_fold[test_idx] = outer_fold
        inner_rows.append(pd.DataFrame({"Fold": outer_fold, "idx": train_idx,
                                        "y": y_outer_train, "prob": best_inner_oof}))

        test_pred = predict_top_fraction(test_probs, positive_fraction)
        test_m = calculate_all_metrics(y_outer_test, test_probs, y_pred=test_pred)
        fold_metrics.append(test_m)

        record = {
            "Semilla": seed,
            "Fold": outer_fold,
            "Epochs": best_candidate["epochs"],
            "Batch": best_candidate["batch_size"],
            "LR": best_candidate["lr"],
            "Weight": best_candidate["weight_mode"],
            "WeightDecay": best_candidate["weight_decay"],
            "Dropout": best_candidate["dropout"],
            f"Inner_{INNER_METRIC}": best_inner_score,
            "Umbral_interno": threshold,
            "Pct_positivos": positive_fraction,
        }
        record.update({name: test_m[key] for key, name in REPORT_METRICS.items()})
        record["Precision"] = test_m["precision"]
        folds_records.append(record)

        print(f" ep={best_candidate['epochs']} | positivos={100 * positive_fraction:.0f}% | AUC-PR={test_m['auc_pr']:.3f} | "
              f"Sens={test_m['sensitivity']:.3f} | Espec={test_m['specificity']:.3f}")

    return folds_records, fold_metrics, oof_probs, oof_fold, pd.concat(inner_rows, ignore_index=True)


def run_or_load_repetition(x, y, dataset, seed, device):
    """Ejecuta una repetición o la carga de disco si ya se había terminado antes."""
    folds_path = os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_folds.csv")
    oof_path = os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_oof.csv")

    if os.path.exists(folds_path) and os.path.exists(oof_path):
        print(f"  [Semilla {seed}] ya calculada, se carga de '{REPS_DIR}/'")
        df_folds = pd.read_csv(folds_path)
        oof = pd.read_csv(oof_path)["prob"].values
        return df_folds, oof

    folds_records, _, oof, oof_fold, df_inner = nested_cv_mlp_random_search(x, y, seed, device=device)
    df_folds = pd.DataFrame(folds_records)
    os.makedirs(REPS_DIR, exist_ok=True)
    df_folds.to_csv(folds_path, index=False)
    pd.DataFrame({"prob": oof, "fold": oof_fold}).to_csv(oof_path, index=False)
    df_inner.to_csv(os.path.join(REPS_DIR, f"{MODEL_TAG}_{dataset}_seed{seed}_inner.csv"), index=False)
    return df_folds, oof


def fmt(mean, std, decimals=3):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["GO", "PathDip"])
    parser.add_argument("--repeats", type=int, default=len(SEEDS), help="nº de repeticiones (semillas) a usar")
    parser.add_argument("--umbral", choices=["gmean", "f1"], default=THRESHOLD_CRITERION,
                        help="métrica que maximiza el umbral de decisión en el bucle interno")
    args = parser.parse_args()

    THRESHOLD_CRITERION = args.umbral
    RESULTS_DIR = results_dir_for(THRESHOLD_CRITERION)
    REPS_DIR = os.path.join(RESULTS_DIR, "reps")
    seeds = SEEDS[: args.repeats]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== MLP BASELINE: VALIDACIÓN CRUZADA ANIDADA REPETIDA ===")
    print(f"Dispositivo: {device} | Repeticiones: {len(seeds)} | Folds: {N_OUTER_SPLITS}x{N_INNER_SPLITS} | "
          f"Candidatos por fold: {N_ITER_SEARCH} | Métrica interna: {INNER_METRIC} | Umbral por: {THRESHOLD_CRITERION}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows, memoria_rows = [], []

    for dataset in args.datasets:
        if not os.path.exists(f"./Data/{dataset}.csv"):
            print(f"[AVISO] No se encontró ./Data/{dataset}.csv, omitiendo...")
            continue

        x, y, gene_names = load_data(dataset)
        all_folds, rep_rows, oof_matrix = [], [], []

        for r, seed in enumerate(seeds, 1):
            print(f"\n  ===== {dataset} | Repetición {r}/{len(seeds)} (semilla {seed}) =====")
            df_folds, oof = run_or_load_repetition(x, y, dataset, seed, device)
            all_folds.append(df_folds.assign(Repeticion=r))
            oof_matrix.append(oof)

            # Un valor por repetición: media de los folds externos
            rep_row = {"Repeticion": r, "Semilla": seed}
            rep_row.update({name: df_folds[name].mean() for name in REPORT_METRICS.values()})
            rep_rows.append(rep_row)
            print(f"  -> Repetición {r}: AUC-ROC={rep_row['AUC-ROC']:.3f} | AUC-PR={rep_row['AUC-PR']:.3f} | "
                  f"G-Mean={rep_row['G-Mean']:.3f} | Sens={rep_row['Sensitivity']:.3f} | Espec={rep_row['Specificity']:.3f}")

        # Ficheros por dataset 
        df_all_folds = pd.concat(all_folds, ignore_index=True)
        df_all_folds.to_csv(os.path.join(RESULTS_DIR, f"metricas_detalladas_folds_{MODEL_TAG}_{dataset}.csv"), index=False)

        df_reps = pd.DataFrame(rep_rows)
        df_reps.to_csv(os.path.join(RESULTS_DIR, f"metricas_por_repeticion_{MODEL_TAG}_{dataset}.csv"), index=False)

        # Probabilidad media de cada gen sobre las repeticiones -> ranking global de candidatos
        oof_matrix = np.vstack(oof_matrix)
        df_rank = pd.DataFrame({"Gen": gene_names, "Clase_CR": y, "Prob_media": oof_matrix.mean(axis=0)})
        for r, seed in enumerate(seeds, 1):
            df_rank[f"Prob_seed{seed}"] = oof_matrix[r - 1]
        df_rank = df_rank.sort_values("Prob_media", ascending=False)
        df_rank.to_csv(os.path.join(RESULTS_DIR, f"ranking_genes_{MODEL_TAG}_{dataset}.csv"), index=False)

        # Gráfico de barras (media sobre repeticiones y folds)
        runs_metrics = [
            [{key: row[name] for key, name in REPORT_METRICS.items()} for _, row in df.iterrows()]
            for df in all_folds
        ]
        generate_evaluation_plots(runs_metrics, output_dir=os.path.join(RESULTS_DIR, f"plots_{dataset}"),
                                  dataset_name=dataset, model_name="MLP_Baseline")

        # Media ± desviación típica entre repeticiones 
        means = df_reps[list(REPORT_METRICS.values())].mean()
        stds = df_reps[list(REPORT_METRICS.values())].std(ddof=1) if len(df_reps) > 1 else means * 0

        summary_rows.append({
            "Fase": "Baseline MLP",
            "Modelo": f"MLP ({dataset})",
            "Nº Feats": "Todas",
            "G-Mean": fmt(means["G-Mean"], stds["G-Mean"]),
            "F1-Score": fmt(means["F1-Score"], stds["F1-Score"]),
            "AUC-ROC": fmt(means["AUC-ROC"], stds["AUC-ROC"]),
            "AUC-PR": fmt(means["AUC-PR"], stds["AUC-PR"]),
        })
        memoria_row = {"Método": "MLP (sin selección)", "Dataset": dataset}
        memoria_row.update({name: fmt(means[name], stds[name]) for name in REPORT_METRICS.values()})
        memoria_rows.append(memoria_row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "tabla_resumen_MLP_NestedCV.csv"), index=False)
        generate_summary_table_image(summary_rows, os.path.join(RESULTS_DIR, "tabla_resumen_MLP_NestedCV.png"))

        df_memoria = pd.DataFrame(memoria_rows)
        df_memoria.to_csv(os.path.join(RESULTS_DIR, "tabla_baseline_media_std.csv"), index=False)

        print("\n==========================================================================")
        print(f"RESULTADOS: media ± desviación típica entre {len(seeds)} repeticiones")
        print("==========================================================================")
        print(df_memoria.to_string(index=False))
        print(f"\n[OK] Ficheros guardados en '{RESULTS_DIR}/'.")
