import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def generate_evaluation_plots(run_metrics_list, output_dir, dataset_name="Dataset", model_name="MLP"):
    os.makedirs(output_dir, exist_ok=True)
    metric_keys = list(run_metrics_list[0][0].keys())
    data_tensor = np.array([[[fold[m] for m in metric_keys] for fold in run] for run in run_metrics_list])
    means = {m: np.mean(data_tensor[:, :, i]) for i, m in enumerate(metric_keys)}

    label_map = {
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1-Score",
        "gmean": "G-Mean",
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "recall_at_10": "Recall@10",
        "ndcg_at_10": "NDCG@10",
        "precision_at_10": "Precision@10",
    }

    labels = [label_map[k] for k in label_map.keys() if k in means]
    values = [means[k] for k in label_map.keys() if k in means]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    colors = sns.color_palette("viridis", len(values))
    bars = ax.bar(labels, values, color=colors, width=0.65, edgecolor="#2c3e50", alpha=0.9)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Puntuación (0-1)", fontsize=12)
    ax.set_xlabel("Métricas de Evaluación", fontsize=12, labelpad=10)
    ax.set_title(f"Métricas del Modelo {model_name} (todas las características) - Dataset {dataset_name}", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=25, ha="right", fontsize=10)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{model_name}_kTodas_Completo.png"))
    plt.close()

def generate_summary_table_image(results_list, output_path):
    """Tabla resumen. Columnas: Fase, Modelo, Nº Feats, G-Mean, F1-Score, AUC-ROC, AUC-PR."""
    df = pd.DataFrame(results_list)
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1.5), dpi=300)
    ax.axis("tight")
    ax.axis("off")

    headers = ["Fase", "Modelo", "Nº Feats", "G-Mean", "F1-Score", "AUC-ROC", "AUC-PR"]
    def cell(v):
        # Acepta números o textos ya formateados como "0.768 ± 0.012"
        return v if isinstance(v, str) else f"{v:.4f}"

    table_data = [
        [
            row["Fase"],
            row["Modelo"],
            row["Nº Feats"],
            cell(row["G-Mean"]),
            cell(row["F1-Score"]),
            cell(row["AUC-ROC"]),
            cell(row["AUC-PR"]),
        ]
        for _, row in df.iterrows()
    ]

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(headers))))  # ancho según el texto de cada columna
    table.scale(1.0, 1.8)

    for i in range(len(headers)):
        cell = table[0, i]
        cell.set_facecolor("#327379")
        cell.set_text_props(color="white", weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
