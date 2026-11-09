"""
Comprobación de sobreajuste / infraajuste del MLP baseline.

Para unas pocas particiones externas, separa un 20 % del entrenamiento como
validación, entrena el MLP durante muchas épocas y registra en cada época la
pérdida y las métricas (AUC-PR y AUC-ROC) en entrenamiento y en validación.
Esto NO toca los datos de test de la validación cruzada anidada.

Uso (desde la raíz del proyecto, donde están Code/ y Data/):
    python check_overfitting.py
    python check_overfitting.py --epochs 150 --folds 5 --lrs 0.0005 0.001
    python check_overfitting.py --dropout 0.5 --weight-decay 1e-2   # config. más regularizada
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from Code.model import SimpleMLP

try:
    from train_mlp import SEED, load_data, set_seed
except ImportError:
    from Code.train_mlp import SEED, load_data, set_seed


def make_criterion(y_tr, weight_mode, device):
    n_pos = np.sum(y_tr)
    n_neg = len(y_tr) - n_pos
    if weight_mode == "moderate":
        w = np.sqrt(n_neg / max(n_pos, 1))
    elif weight_mode == "full":
        w = n_neg / max(n_pos, 1)
    else:
        w = 1.0
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], dtype=torch.float32, device=device))


def eval_split(model, x, y, criterion, device):
    """Pérdida, AUC-ROC y AUC-PR en modo evaluación (sin dropout)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        target = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        loss = criterion(logits, target).item()
        p = torch.sigmoid(logits).cpu().numpy().ravel()
    roc = roc_auc_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return loss, roc, auc(rec, prec)


def run_curve(x_tr, y_tr, x_val, y_val, lr, batch_size, weight_mode, epochs, seed, device,
              dropout=0.2, weight_decay=1e-4):
    set_seed(seed)
    model = SimpleMLP(x_tr.shape[1], hidden_dim1=50, hidden_dim2=25, dropout_rate=dropout).to(device)
    criterion = make_criterion(y_tr, weight_mode, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=True, generator=gen,
    )

    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        tr_loss, tr_roc, tr_pr = eval_split(model, x_tr, y_tr, criterion, device)
        va_loss, va_roc, va_pr = eval_split(model, x_val, y_val, criterion, device)
        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss, "val_loss": va_loss,
            "train_auc_roc": tr_roc, "val_auc_roc": va_roc,
            "train_auc_pr": tr_pr, "val_auc_pr": va_pr,
        })
    return pd.DataFrame(rows)


def summarize(mean_df, dataset, lr, ref_epochs=(40, 50)):
    """Imprime un resumen orientativo de las curvas medias."""
    best_loss_ep = int(mean_df.loc[mean_df["val_loss"].idxmin(), "epoch"])
    best_pr_ep = int(mean_df.loc[mean_df["val_auc_pr"].idxmax(), "epoch"])
    print(f"\n--- {dataset} | lr={lr} ---")
    print(f"  Mínima pérdida de validación en la época {best_loss_ep} "
          f"({mean_df.loc[best_loss_ep - 1, 'val_loss']:.4f})")
    print(f"  Máximo AUC-PR de validación en la época {best_pr_ep} "
          f"({mean_df.loc[best_pr_ep - 1, 'val_auc_pr']:.4f})")
    for ep in ref_epochs:
        r = mean_df.loc[ep - 1]
        print(f"  Época {ep}: pérdida train={r.train_loss:.4f} val={r.val_loss:.4f} | "
              f"AUC-PR train={r.train_auc_pr:.3f} val={r.val_auc_pr:.3f}")

    r50 = mean_df.loc[50 - 1]
    min_val = mean_df["val_loss"].min()
    if best_loss_ep < 40 and r50.val_loss > 1.05 * min_val:
        verdict = ("Indicios de SOBREAJUSTE: la pérdida de validación toca fondo antes de la época 40 "
                   "y sube después, mientras la de entrenamiento sigue bajando.")
    elif best_loss_ep >= 45 and r50.train_auc_pr < 0.7:
        verdict = ("Indicios de INFRAAJUSTE: la validación todavía mejora hacia la época 50 "
                   "y el modelo no ajusta ni siquiera el entrenamiento.")
    else:
        verdict = "Sin señales claras de sobreajuste ni de infraajuste (criterio orientativo)."
    print("  " + verdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["GO", "PathDip"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3, help="nº de particiones externas a usar")
    parser.add_argument("--lrs", nargs="+", type=float, default=[5e-4, 1e-3, 2e-3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-mode", default="moderate")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default="results/overfitting_check")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    tag = f"do{args.dropout}_wd{args.weight_decay}"
    print(f"Configuración: dropout={args.dropout} | weight_decay={args.weight_decay}")

    for dataset in args.datasets:
        if not os.path.exists(f"./Data/{dataset}.csv"):
            print(f"[AVISO] No se encontró ./Data/{dataset}.csv, omitiendo...")
            continue
        x, y, _ = load_data(dataset)
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        splits = list(outer_cv.split(x, y))[: args.folds]

        for lr in args.lrs:
            all_curves = []
            for f_idx, (train_idx, _test_idx) in enumerate(splits, 1):
                x_outer, y_outer = x[train_idx], y[train_idx]
                # Validación interna: el test externo no se usa en ningún momento.
                x_tr, x_val, y_tr, y_val = train_test_split(
                    x_outer, y_outer, test_size=0.2, stratify=y_outer, random_state=SEED + f_idx
                )
                curve = run_curve(x_tr, y_tr, x_val, y_val, lr, args.batch_size,
                                  args.weight_mode, args.epochs, SEED + f_idx, device,
                                  args.dropout, args.weight_decay)
                curve["fold"] = f_idx
                all_curves.append(curve)
                print(f"[{dataset}] lr={lr} fold {f_idx}/{len(splits)} listo")

            curves = pd.concat(all_curves, ignore_index=True)
            curves.insert(0, "lr", lr)
            curves.insert(0, "dataset", dataset)
            curves.to_csv(os.path.join(args.out, f"curvas_{dataset}_lr{lr}_{tag}.csv"), index=False)

            mean_df = curves.groupby("epoch").mean(numeric_only=True).reset_index()
            std_df = curves.groupby("epoch").std(numeric_only=True).reset_index()
            summarize(mean_df, dataset, lr)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=200)
            for ax, (cols, title) in zip(
                axes,
                [(("train_loss", "val_loss"), "Pérdida ponderada (BCE)"),
                 (("train_auc_pr", "val_auc_pr"), "AUC-PR")],
            ):
                for col, label, color in zip(cols, ("Entrenamiento", "Validación"), ("#327379", "#b10072")):
                    ax.plot(mean_df["epoch"], mean_df[col], label=label, color=color)
                    ax.fill_between(mean_df["epoch"], mean_df[col] - std_df[col],
                                    mean_df[col] + std_df[col], color=color, alpha=0.12)
                for ep in (40, 50):
                    ax.axvline(ep, color="gray", linestyle="--", linewidth=0.8)
                ax.set_xlabel("Época")
                ax.set_title(f"{title} - {dataset} (lr={lr}, dropout={args.dropout}, wd={args.weight_decay})", fontsize=9)
                ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"curvas_{dataset}_lr{lr}_{tag}.png"))
            plt.close()

    print(f"\n[OK] Curvas y CSV guardados en '{args.out}/'")


if __name__ == "__main__":
    main()
