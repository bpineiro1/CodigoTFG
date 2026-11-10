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

Inferencia Monte Carlo:
  En entrenamiento el DDS muestrea las compuertas con ruido (selecciones aleatorias), pero
  en la versión 1 la evaluación era determinista (sin ruido, Top-K fijo). Ahora cada modelo
  entrenado se evalúa de dos formas:
    - det: inferencia determinista, igual que en la versión 1.
    - mc:  dropout apagado, muestreo del DDS activo y media de N_MC probabilidades por gen.
  El entrenamiento no cambia, así que las dos inferencias se comparan sobre exactamente los
  mismos modelos entrenados. Cada una elige su propio candidato y su propio umbral en el
  bucle interno, con su misma forma de inferencia. La parte "det" debe reproducir los
  resultados de la versión 1, lo que sirve de comprobación.

Además del rendimiento, guarda cuántas características usa cada gen y con qué
frecuencia se selecciona cada característica en genes CR y NotCR.

Uso (desde la raíz del proyecto, junto a train_mlp.py):
    python train_dds.py                      # 10 repeticiones, GO y PathDip
    python train_dds.py --repeats 1 --datasets PathDip
Resultados en results/mc/ (los de la versión 1 en results/ no se tocan).
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
    PARAM_SPACE, REPORT_METRICS, RESULTS_DIR, set_seed, fmt,
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

# Inferencia
N_MC = 30                  # nº de pasadas Monte Carlo que se promedian por gen
MC_SEED_OFFSET = 500_000   # semilla del muestreo en evaluación = semilla del modelo + offset
MODES = ("det", "mc")
MODE_TAG = {"det": "MLP_DDS_DET", "mc": "MLP_DDS_MC"}
MODE_NAME = {"det": "MLP + DDS (determinista)", "mc": f"MLP + DDS (Monte Carlo, N={N_MC})"}

DDS_RESULTS_DIR = os.path.join(RESULTS_DIR, "mc")
DDS_REPS_DIR = os.path.join(DDS_RESULTS_DIR, "reps")


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


def evaluate_dds_model(model, x_eval, device="cpu", n_mc=0, mc_seed=0):
    """
    Devuelve probabilidades, máscaras de selección (solo variables activas) y nº de variables usadas.
      n_mc = 0: inferencia determinista (sin ruido en las compuertas), como en la versión 1.
      n_mc > 0: inferencia Monte Carlo: dropout apagado (model.eval()) pero con el muestreo
                del DDS activo; se promedian las probabilidades de n_mc pasadas. Las máscaras
                y el nº de variables usadas también se promedian (fracción de pasadas en que
                se seleccionó cada variable).
    """
    model.eval()
    x_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        if n_mc <= 0:
            logits, _, hard_mask, n_used = model(x_t)
            return (torch.sigmoid(logits).cpu().numpy().flatten(),
                    hard_mask.cpu().numpy(), n_used.cpu().numpy())

        torch.manual_seed(mc_seed)  # muestreo reproducible
        model.dds.mc_sampling = True
        try:
            probs_sum, masks_sum, used_sum = 0.0, 0.0, 0.0
            for _ in range(n_mc):
                logits, _, hard_mask, n_used = model(x_t)
                probs_sum = probs_sum + torch.sigmoid(logits)
                masks_sum = masks_sum + hard_mask
                used_sum = used_sum + n_used
        finally:
            model.dds.mc_sampling = False

    return ((probs_sum / n_mc).cpu().numpy().flatten(),
            (masks_sum / n_mc).cpu().numpy(), (used_sum / n_mc).cpu().numpy())


def n_mc_for(mode):
    return N_MC if mode == "mc" else 0


def nested_cv_dds(x, y, dataset, seed, device="cpu"):
    """
    Una repetición de la validación cruzada anidada con la semilla seed.
    Devuelve, para cada forma de inferencia (det / mc): registros por fold, probabilidades
    out-of-fold y conteos de selección por característica.
    """
    outer_cv = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed % (2**32))
    n_feat = x.shape[1]
    out = {
        m: {"records": [], "oof": np.full(len(y), np.nan), "sel": {1: np.zeros(n_feat), 0: np.zeros(n_feat)}}
        for m in MODES
    }
    act_counts = {1: np.zeros(n_feat), 0: np.zeros(n_feat)}

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), 1):
        x_outer_train, x_outer_test = x[train_idx], x[test_idx]
        y_outer_train, y_outer_test = y[train_idx], y[test_idx]

        # Bucle interno: Random Search (cada modelo se evalúa con las dos inferencias)
        inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=(seed + outer_fold) % (2**32))
        candidates = sample_candidates(seed, outer_fold, dataset)
        best = {m: {"score": -1.0, "idx": None, "oof": None} for m in MODES}
        print(f"    Fold {outer_fold:02d}/{N_OUTER_SPLITS} ", end="", flush=True)

        for cand_idx, cand in enumerate(candidates):
            inner_scores = {m: [] for m in MODES}
            inner_oof = {m: np.zeros(len(y_outer_train)) for m in MODES}
            for in_fold, (in_tr_idx, in_val_idx) in enumerate(inner_cv.split(x_outer_train, y_outer_train)):
                train_seed = seed + outer_fold * 1000 + cand_idx * 10 + in_fold
                inner_model = train_single_dds_model(
                    x_outer_train[in_tr_idx], y_outer_train[in_tr_idx], cand, seed=train_seed, device=device
                )
                for m in MODES:
                    val_probs, _, _ = evaluate_dds_model(
                        inner_model, x_outer_train[in_val_idx], device=device,
                        n_mc=n_mc_for(m), mc_seed=train_seed + MC_SEED_OFFSET,
                    )
                    inner_oof[m][in_val_idx] = val_probs
                    inner_scores[m].append(calculate_all_metrics(y_outer_train[in_val_idx], val_probs)[INNER_METRIC])
            print(".", end="", flush=True)

            for m in MODES:
                mean_score = float(np.mean(inner_scores[m]))
                if mean_score > best[m]["score"]:
                    best[m] = {"score": mean_score, "idx": cand_idx, "oof": inner_oof[m]}

        # Reentrenamiento final y test. Si las dos inferencias eligen el mismo candidato,
        # el modelo final es el mismo (misma semilla) y se entrena una sola vez.
        final_seed = seed + outer_fold * 1000 + 999
        final_models = {}
        n_active = (x_outer_test > 0).sum(axis=1)
        for cls in (0, 1):
            act_counts[cls] += (x_outer_test[y_outer_test == cls] > 0).sum(axis=0)

        fold_txt = []
        for m in MODES:
            cand_idx = best[m]["idx"]
            cand = candidates[cand_idx]

            # Umbral: igual que en el baseline (G-Mean en el bucle interno, trasladado como porcentaje)
            threshold = best_threshold(y_outer_train, best[m]["oof"], criterion=THRESHOLD_CRITERION)
            positive_fraction = float(np.mean(best[m]["oof"] >= threshold))

            if cand_idx not in final_models:
                final_models[cand_idx] = train_single_dds_model(
                    x_outer_train, y_outer_train, cand, seed=final_seed, device=device)
            test_probs, test_masks, test_n_used = evaluate_dds_model(
                final_models[cand_idx], x_outer_test, device=device,
                n_mc=n_mc_for(m), mc_seed=final_seed + MC_SEED_OFFSET,
            )
            out[m]["oof"][test_idx] = test_probs

            test_pred = predict_top_fraction(test_probs, positive_fraction)
            test_m = calculate_all_metrics(y_outer_test, test_probs, y_pred=test_pred)

            for cls in (0, 1):
                out[m]["sel"][cls] += test_masks[y_outer_test == cls].sum(axis=0)

            record = {
                "Semilla": seed,
                "Fold": outer_fold,
                "Inferencia": m,
                "N_MC": n_mc_for(m),
                "K": cand["k_features"],
                "Penalty": cand["penalty_weight"],
                "Epochs": cand["epochs"],
                "Batch": cand["batch_size"],
                "LR": cand["lr"],
                "Weight": cand["weight_mode"],
                "WeightDecay": cand["weight_decay"],
                "Dropout": cand["dropout"],
                f"Inner_{INNER_METRIC}": best[m]["score"],
                "Umbral_interno": threshold,
                "Pct_positivos": positive_fraction,
                "Feats_activas_media": float(n_active.mean()),
                "Feats_usadas_media": float(test_n_used.mean()),
                "Pct_activas_usadas": float(100 * test_n_used.sum() / max(n_active.sum(), 1)),
            }
            record.update({name: test_m[key] for key, name in REPORT_METRICS.items()})
            record["Precision"] = test_m["precision"]
            out[m]["records"].append(record)
            fold_txt.append(f"{m.upper()}: K={cand['k_features']} AUC-PR={test_m['auc_pr']:.3f} "
                            f"Sens={test_m['sensitivity']:.2f} Espec={test_m['specificity']:.2f}")

        print(f" usa ~{out['mc']['records'][-1]['Feats_usadas_media']:.0f} de {n_active.mean():.0f} activas | "
              + " | ".join(fold_txt))

    results = {}
    for m in MODES:
        df_freq = pd.DataFrame({
            "Sel_CR": out[m]["sel"][1], "Act_CR": act_counts[1],
            "Sel_NotCR": out[m]["sel"][0], "Act_NotCR": act_counts[0],
        })
        results[m] = (out[m]["records"], out[m]["oof"], df_freq)
    return results


def rep_paths(dataset, seed, mode):
    tag = MODE_TAG[mode]
    return {k: os.path.join(DDS_REPS_DIR, f"{tag}_{dataset}_seed{seed}_{k}.csv") for k in ("folds", "oof", "freq")}


def run_or_load_repetition(x, y, dataset, seed, device):
    """Ejecuta una repetición (las dos inferencias) o la carga de disco si ya se había terminado antes."""
    paths = {m: rep_paths(dataset, seed, m) for m in MODES}
    if all(os.path.exists(p) for m in MODES for p in paths[m].values()):
        print(f"  [Semilla {seed}] ya calculada, se carga de '{DDS_REPS_DIR}/'")
        return {m: (pd.read_csv(paths[m]["folds"]), pd.read_csv(paths[m]["oof"])["prob"].values,
                    pd.read_csv(paths[m]["freq"])) for m in MODES}

    results = nested_cv_dds(x, y, dataset, seed, device=device)
    os.makedirs(DDS_REPS_DIR, exist_ok=True)
    loaded = {}
    for m, (records, oof, df_freq) in results.items():
        df_folds = pd.DataFrame(records)
        df_folds.to_csv(paths[m]["folds"], index=False)
        pd.DataFrame({"prob": oof}).to_csv(paths[m]["oof"], index=False)
        df_freq.to_csv(paths[m]["freq"], index=False)
        loaded[m] = (df_folds, oof, df_freq)
    return loaded


def save_merged(rows, path, keys):
    """Guarda la tabla conservando las filas de otros datasets que ya hubiera en el fichero."""
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        if all(k in df_old.columns for k in keys):
            new_keys = set(map(tuple, df_new[keys].values))
            df_old = df_old[[tuple(r) not in new_keys for r in df_old[keys].values]]
            df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)
    return df_new


def aggregate_mode(mode, dataset, seeds, folds_list, oof_list, freq_total, y, gene_names, feature_names):
    """Escribe los ficheros de un dataset para una forma de inferencia y devuelve sus filas de resumen."""
    tag = MODE_TAG[mode]

    df_all_folds = pd.concat(folds_list, ignore_index=True)
    df_all_folds.to_csv(os.path.join(DDS_RESULTS_DIR, f"metricas_detalladas_folds_{tag}_{dataset}.csv"), index=False)

    rep_rows = []
    for r, (seed, df) in enumerate(zip(seeds, folds_list), 1):
        row = {"Repeticion": r, "Semilla": seed}
        row.update({name: df[name].mean() for name in REPORT_METRICS.values()})
        row["Feats_usadas_media"] = df["Feats_usadas_media"].mean()
        row["Pct_activas_usadas"] = df["Pct_activas_usadas"].mean()
        rep_rows.append(row)
    df_reps = pd.DataFrame(rep_rows)
    df_reps.to_csv(os.path.join(DDS_RESULTS_DIR, f"metricas_por_repeticion_{tag}_{dataset}.csv"), index=False)

    oof_matrix = np.vstack(oof_list)
    df_rank = pd.DataFrame({"Gen": gene_names, "Clase_CR": y, "Prob_media": oof_matrix.mean(axis=0)})
    for r, seed in enumerate(seeds, 1):
        df_rank[f"Prob_seed{seed}"] = oof_matrix[r - 1]
    df_rank.sort_values("Prob_media", ascending=False).to_csv(
        os.path.join(DDS_RESULTS_DIR, f"ranking_genes_{tag}_{dataset}.csv"), index=False)

    # Frecuencia de selección de cada característica (sobre todas las repeticiones):
    #   Sel_%: en qué % de los genes de la clase se seleccionó (en MC, media de las pasadas)
    #   Sel_si_activa_%: de los genes de la clase que la tienen activa, en qué % se seleccionó
    df_feat = pd.DataFrame({"Caracteristica": feature_names})
    n_cls = {"CR": len(seeds) * int(np.sum(y == 1)), "NotCR": len(seeds) * int(np.sum(y == 0))}
    for cls in ("CR", "NotCR"):
        df_feat[f"Sel_{cls}_%"] = 100 * freq_total[f"Sel_{cls}"].values / n_cls[cls]
        df_feat[f"Activa_{cls}_%"] = 100 * freq_total[f"Act_{cls}"].values / n_cls[cls]
        df_feat[f"Sel_si_activa_{cls}_%"] = 100 * freq_total[f"Sel_{cls}"].values / np.maximum(freq_total[f"Act_{cls}"].values, 1)
    df_feat["Diferencia_CR_menos_NotCR"] = df_feat["Sel_CR_%"] - df_feat["Sel_NotCR_%"]
    df_feat = df_feat.sort_values("Diferencia_CR_menos_NotCR", ascending=False)
    df_feat.round(2).to_csv(os.path.join(DDS_RESULTS_DIR, f"frecuencia_seleccion_{tag}_{dataset}.csv"), index=False)

    runs_metrics = [
        [{key: row[name] for key, name in REPORT_METRICS.items()} for _, row in df.iterrows()]
        for df in folds_list
    ]
    generate_evaluation_plots(runs_metrics, output_dir=os.path.join(DDS_RESULTS_DIR, f"plots_{dataset}_{mode}"),
                              dataset_name=dataset, model_name=tag)

    # Media ± desviación típica entre repeticiones
    cols = list(REPORT_METRICS.values())
    means = df_reps[cols].mean()
    stds = df_reps[cols].std(ddof=1) if len(df_reps) > 1 else means * 0
    feats_txt = (f"{df_reps['Feats_usadas_media'].mean():.0f} por gen "
                 f"({df_reps['Pct_activas_usadas'].mean():.0f}% de las activas)")

    summary_row = {
        "Fase": "Determinista" if mode == "det" else f"Monte Carlo (N={N_MC})",
        "Modelo": f"MLP + DDS ({dataset})",
        "Nº Feats": feats_txt,
        "G-Mean": fmt(means["G-Mean"], stds["G-Mean"]),
        "F1-Score": fmt(means["F1-Score"], stds["F1-Score"]),
        "AUC-ROC": fmt(means["AUC-ROC"], stds["AUC-ROC"]),
        "AUC-PR": fmt(means["AUC-PR"], stds["AUC-PR"]),
    }
    memoria_row = {"Método": MODE_NAME[mode], "Dataset": dataset, "Nº Feats": feats_txt}
    memoria_row.update({name: fmt(means[name], stds[name]) for name in cols})
    return summary_row, memoria_row, df_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["GO", "PathDip"])
    parser.add_argument("--repeats", type=int, default=len(SEEDS), help="nº de repeticiones (semillas) a usar")
    args = parser.parse_args()

    seeds = SEEDS[: args.repeats]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== MLP + DDS: VALIDACIÓN CRUZADA ANIDADA REPETIDA (inferencia determinista y Monte Carlo) ===")
    print(f"Dispositivo: {device} | Repeticiones: {len(seeds)} | Folds: {N_OUTER_SPLITS}x{N_INNER_SPLITS} | "
          f"Candidatos por fold: {N_ITER_SEARCH} | Métrica interna: {INNER_METRIC} | Umbral por: {THRESHOLD_CRITERION} | "
          f"Pasadas MC: {N_MC}")

    os.makedirs(DDS_RESULTS_DIR, exist_ok=True)
    summary_rows, memoria_rows = [], []

    for dataset in args.datasets:
        if not os.path.exists(f"./Data/{dataset}.csv"):
            print(f"[AVISO] No se encontró ./Data/{dataset}.csv, omitiendo...")
            continue
        if dataset not in K_GRID:
            print(f"[AVISO] No hay rejilla de K definida para {dataset}, omitiendo...")
            continue

        x, y, gene_names, feature_names = load_data(dataset)
        folds = {m: [] for m in MODES}
        oofs = {m: [] for m in MODES}
        freqs = {m: None for m in MODES}

        for r, seed in enumerate(seeds, 1):
            print(f"\n  ===== {dataset} | Repetición {r}/{len(seeds)} (semilla {seed}) =====")
            results = run_or_load_repetition(x, y, dataset, seed, device)
            txt = []
            for m in MODES:
                df_folds, oof, df_freq = results[m]
                folds[m].append(df_folds.assign(Repeticion=r))
                oofs[m].append(oof)
                freqs[m] = df_freq if freqs[m] is None else freqs[m] + df_freq
                txt.append(f"{m.upper()}: AUC-ROC={df_folds['AUC-ROC'].mean():.3f} AUC-PR={df_folds['AUC-PR'].mean():.3f} "
                           f"G-Mean={df_folds['G-Mean'].mean():.3f}")
            print("  -> Repetición " + str(r) + ": " + " | ".join(txt))

        for m in MODES:
            summary_row, memoria_row, df_feat = aggregate_mode(
                m, dataset, seeds, folds[m], oofs[m], freqs[m], y, gene_names, feature_names)
            summary_rows.append(summary_row)
            memoria_rows.append(memoria_row)

        print(f"\n---> Características más seleccionadas en genes CR frente a NotCR ({dataset}, Monte Carlo):")
        print(df_feat.head(10)[["Caracteristica", "Sel_CR_%", "Sel_NotCR_%", "Sel_si_activa_CR_%"]].round(1).to_string(index=False))

    if summary_rows:
        df_summary = save_merged(summary_rows, os.path.join(DDS_RESULTS_DIR, "tabla_resumen_MLP_DDS_NestedCV.csv"),
                                 keys=["Fase", "Modelo"])
        generate_summary_table_image(df_summary.to_dict("records"),
                                     os.path.join(DDS_RESULTS_DIR, "tabla_resumen_MLP_DDS_NestedCV.png"))
        df_memoria = save_merged(memoria_rows, os.path.join(DDS_RESULTS_DIR, "tabla_dds_media_std.csv"),
                                 keys=["Método", "Dataset"])

        print("\n==========================================================================")
        print(f"RESULTADOS: media ± desviación típica entre {len(seeds)} repeticiones")
        print("==========================================================================")
        print(df_memoria.to_string(index=False))
        print(f"\n[OK] Ficheros guardados en '{DDS_RESULTS_DIR}/'.")
        print("Para comparar con el MLP y con la inferencia determinista: python compare_models.py --mc")
