"""
Comparación estadística MLP (baseline) frente a MLP + DDS.

Lee los resultados que generan train_mlp.py y train_dds.py y, para cada dataset y métrica:
  - Media ± desviación típica entre repeticiones de cada modelo.
  - t-test pareado sobre las medias de las repeticiones (como en el TFG de referencia).
  - t-test pareado corregido para validación cruzada repetida (Nadeau y Bengio, 2003;
    Bouckaert y Frank, 2004) sobre los folds emparejados. Es más conservador, porque
    tiene en cuenta que los folds comparten datos de entrenamiento.
Ambos modelos usan exactamente las mismas particiones (mismas semillas), así que cada
fold del MLP se compara con el mismo fold del MLP + DDS.

El símbolo † marca diferencias significativas (p < 0.05) según el test corregido.

Uso (desde la raíz del proyecto, después de ejecutar los dos scripts):
    python compare_models.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results"
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


def main():
    rows, pvals = [], []
    for ds in DATASETS:
        paths = {
            "rep_mlp": os.path.join(RESULTS_DIR, f"metricas_por_repeticion_MLP_{ds}.csv"),
            "rep_dds": os.path.join(RESULTS_DIR, f"metricas_por_repeticion_MLP_DDS_{ds}.csv"),
            "fold_mlp": os.path.join(RESULTS_DIR, f"metricas_detalladas_folds_MLP_{ds}.csv"),
            "fold_dds": os.path.join(RESULTS_DIR, f"metricas_detalladas_folds_MLP_DDS_{ds}.csv"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[AVISO] Faltan resultados de {ds}, se omite.")
            continue

        rep_mlp, rep_dds = pd.read_csv(paths["rep_mlp"]), pd.read_csv(paths["rep_dds"])
        rep = rep_mlp.merge(rep_dds, on="Semilla", suffixes=("_mlp", "_dds"))
        fold = pd.read_csv(paths["fold_mlp"]).merge(
            pd.read_csv(paths["fold_dds"]), on=["Semilla", "Fold"], suffixes=("_mlp", "_dds"))
        n_folds = int(fold["Fold"].max())
        print(f"[{ds}] {len(rep)} repeticiones emparejadas, {len(fold)} folds emparejados")

        row_mlp = {"Método": "MLP (sin selección)", "Dataset": ds}
        row_dds = {"Método": "MLP + DDS", "Dataset": ds}
        for m in METRICS:
            a, b = rep[f"{m}_mlp"].values, rep[f"{m}_dds"].values
            p_rep = float(stats.ttest_rel(b, a).pvalue) if len(rep) > 1 else np.nan
            p_corr = corrected_ttest(fold[f"{m}_dds"].values - fold[f"{m}_mlp"].values, n_folds)

            mark_mlp = mark_dds = ""
            if p_corr < ALPHA:
                if b.mean() > a.mean():
                    mark_dds = " †"
                else:
                    mark_mlp = " †"
            row_mlp[m] = f"{a.mean():.3f} ± {a.std(ddof=1):.3f}{mark_mlp}"
            row_dds[m] = f"{b.mean():.3f} ± {b.std(ddof=1):.3f}{mark_dds}"
            pvals.append({
                "Dataset": ds, "Métrica": m,
                "Media MLP": round(a.mean(), 4), "Media MLP+DDS": round(b.mean(), 4),
                "Diferencia (DDS - MLP)": round(b.mean() - a.mean(), 4),
                "Folds en que gana DDS": f"{int(np.sum(fold[f'{m}_dds'] > fold[f'{m}_mlp']))}/{len(fold)}",
                "p (t pareado, repeticiones)": round(p_rep, 4),
                "p (t corregido, folds)": round(p_corr, 4),
            })
        rows += [row_mlp, row_dds]

    if not rows:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_tab, df_p = pd.DataFrame(rows), pd.DataFrame(pvals)
    df_tab.to_csv(os.path.join(RESULTS_DIR, "tabla_comparativa_MLP_vs_DDS.csv"), index=False)
    df_p.to_csv(os.path.join(RESULTS_DIR, "tests_estadisticos_MLP_vs_DDS.csv"), index=False)

    print("\nTABLA COMPARATIVA (media ± desviación típica entre repeticiones; † = p < 0.05, t corregido)")
    print(df_tab.to_string(index=False))
    print("\nTESTS ESTADÍSTICOS")
    print(df_p.to_string(index=False))


if __name__ == "__main__":
    main()
