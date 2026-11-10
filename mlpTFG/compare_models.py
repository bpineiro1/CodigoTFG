"""
Comparación estadística entre dos modelos evaluados con las mismas particiones.

Para cada dataset y métrica:
  - Media ± desviación típica entre repeticiones de cada modelo.
  - t-test pareado sobre las medias de las repeticiones (como en el TFG de referencia).
  - t-test pareado corregido para validación cruzada repetida 
    sobre los folds emparejados. Es más conservador, porque
    tiene en cuenta que los folds comparten datos de entrenamiento.
Ambos modelos usan exactamente las mismas particiones (mismas semillas), así que cada
fold de un modelo se compara con el mismo fold del otro.

El símbolo † marca diferencias significativas (p < 0.05) según el test corregido.

Uso (desde la raíz del proyecto):
    python compare_models.py         # versión 1: MLP frente a MLP + DDS (results/)
    python compare_models.py --mc    # inferencia Monte Carlo (results/mc/):
                                     #   MLP frente a MLP + DDS (MC) y
                                     #   MLP + DDS determinista frente a MLP + DDS (MC)
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
MC_DIR = os.path.join(RESULTS_DIR, "mc")
DATASETS = ["GO", "PathDip"]
METRICS = ["AUC-ROC", "AUC-PR", "G-Mean", "F1-Score", "NDCG@10", "Precision@10", "Sensitivity", "Specificity"]
ALPHA = 0.05


def corrected_ttest(d, n_folds):
    """t-test pareado corregido para k-fold repetido (diferencias por fold `d`)."""
    n = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 1.0
    test_train_ratio = 1.0 / (n_folds - 1)
    t = np.mean(d) / np.sqrt((1.0 / n + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def compare(a, b, out_dir, out_name):
    """
    Compara el modelo `a` (referencia) con el modelo `b`.
    a, b: diccionarios con dir (carpeta), tag (sufijo de los ficheros) y name (nombre en la tabla).
    """
    rows, pvals = [], []
    for ds in DATASETS:
        paths = {
            "rep_a": os.path.join(a["dir"], f"metricas_por_repeticion_{a['tag']}_{ds}.csv"),
            "rep_b": os.path.join(b["dir"], f"metricas_por_repeticion_{b['tag']}_{ds}.csv"),
            "fold_a": os.path.join(a["dir"], f"metricas_detalladas_folds_{a['tag']}_{ds}.csv"),
            "fold_b": os.path.join(b["dir"], f"metricas_detalladas_folds_{b['tag']}_{ds}.csv"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[AVISO] Faltan resultados de {ds} para {a['name']} frente a {b['name']}, se omite.")
            continue

        rep = pd.read_csv(paths["rep_a"]).merge(pd.read_csv(paths["rep_b"]), on="Semilla", suffixes=("_a", "_b"))
        fold = pd.read_csv(paths["fold_a"]).merge(
            pd.read_csv(paths["fold_b"]), on=["Semilla", "Fold"], suffixes=("_a", "_b"))
        n_folds = int(fold["Fold"].max())
        print(f"[{ds}] {len(rep)} repeticiones emparejadas, {len(fold)} folds emparejados")

        row_a = {"Método": a["name"], "Dataset": ds}
        row_b = {"Método": b["name"], "Dataset": ds}
        for m in METRICS:
            va, vb = rep[f"{m}_a"].values, rep[f"{m}_b"].values
            p_rep = float(stats.ttest_rel(vb, va).pvalue) if len(rep) > 1 else np.nan
            p_corr = corrected_ttest(fold[f"{m}_b"].values - fold[f"{m}_a"].values, n_folds)

            mark_a = mark_b = ""
            if p_corr < ALPHA:
                if vb.mean() > va.mean():
                    mark_b = " †"
                else:
                    mark_a = " †"
            sd_a = va.std(ddof=1) if len(va) > 1 else 0.0
            sd_b = vb.std(ddof=1) if len(vb) > 1 else 0.0
            row_a[m] = f"{va.mean():.3f} ± {sd_a:.3f}{mark_a}"
            row_b[m] = f"{vb.mean():.3f} ± {sd_b:.3f}{mark_b}"
            pvals.append({
                "Dataset": ds, "Métrica": m,
                f"Media {a['name']}": round(va.mean(), 4), f"Media {b['name']}": round(vb.mean(), 4),
                "Diferencia (B - A)": round(vb.mean() - va.mean(), 4),
                "Folds en que gana B": f"{int(np.sum(fold[f'{m}_b'] > fold[f'{m}_a']))}/{len(fold)}",
                "p (t pareado, repeticiones)": round(p_rep, 4),
                "p (t corregido, folds)": round(p_corr, 4),
            })
        rows += [row_a, row_b]

    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    df_tab, df_p = pd.DataFrame(rows), pd.DataFrame(pvals)
    df_tab.to_csv(os.path.join(out_dir, f"tabla_comparativa_{out_name}.csv"), index=False)
    df_p.to_csv(os.path.join(out_dir, f"tests_estadisticos_{out_name}.csv"), index=False)

    print(f"\nTABLA COMPARATIVA: {a['name']} (A) frente a {b['name']} (B)")
    print("(media ± desviación típica entre repeticiones; † = p < 0.05, t corregido)")
    print(df_tab.to_string(index=False))
    print("\nTESTS ESTADÍSTICOS")
    print(df_p.to_string(index=False))


def check_det_reproduces_v1():
    """La inferencia determinista de results/mc debe dar lo mismo que la versión 1 (results/)."""
    for ds in DATASETS:
        p_v1 = os.path.join(RESULTS_DIR, f"metricas_detalladas_folds_MLP_DDS_{ds}.csv")
        p_det = os.path.join(MC_DIR, f"metricas_detalladas_folds_MLP_DDS_DET_{ds}.csv")
        if not (os.path.exists(p_v1) and os.path.exists(p_det)):
            continue
        v1 = pd.read_csv(p_v1).merge(pd.read_csv(p_det), on=["Semilla", "Fold"], suffixes=("_v1", "_det"))
        diff = max(float(np.max(np.abs(v1[f"{m}_v1"] - v1[f"{m}_det"]))) for m in ["AUC-ROC", "AUC-PR", "G-Mean", "F1-Score"])
        estado = "OK, reproduce la versión 1" if diff < 1e-6 else "AVISO: no coincide exactamente con la versión 1"
        print(f"[Comprobación {ds}] {len(v1)} folds comunes, diferencia máxima con la v1 = {diff:.6f} -> {estado}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", action="store_true", help="comparar la inferencia Monte Carlo (results/mc)")
    args = parser.parse_args()

    mlp = {"dir": RESULTS_DIR, "tag": "MLP", "name": "MLP (sin selección)"}
    if not args.mc:
        dds = {"dir": RESULTS_DIR, "tag": "MLP_DDS", "name": "MLP + DDS"}
        compare(mlp, dds, RESULTS_DIR, "MLP_vs_DDS")
    else:
        det = {"dir": MC_DIR, "tag": "MLP_DDS_DET", "name": "MLP + DDS (determinista)"}
        mc = {"dir": MC_DIR, "tag": "MLP_DDS_MC", "name": "MLP + DDS (Monte Carlo)"}
        check_det_reproduces_v1()
        print("\n" + "=" * 90)
        compare(mlp, mc, MC_DIR, "MLP_vs_DDS_MC")
        print("\n" + "=" * 90)
        compare(det, mc, MC_DIR, "DDS_det_vs_DDS_MC")
