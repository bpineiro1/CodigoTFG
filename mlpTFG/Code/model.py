import torch
import torch.nn as nn

from Code.dds import DynamicDataSelectionHard2


def _mlp_block(input_dim: int, hidden_dim1: int, hidden_dim2: int, dropout_rate: float) -> nn.Sequential:
    """Bloque clasificador común: 50 -> 25 -> 10 -> 1 con LayerNorm, SiLU y Dropout."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.LayerNorm(hidden_dim1),
        nn.SiLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.LayerNorm(hidden_dim2),
        nn.SiLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim2, 10),
        nn.LayerNorm(10),
        nn.SiLU(),
        nn.Linear(10, 1),  # logit
    )


class SimpleMLP(nn.Module):
    """MLP baseline: usa todas las características."""

    def __init__(self, input_dim: int, hidden_dim1: int = 50, hidden_dim2: int = 25, dropout_rate: float = 0.2):
        super().__init__()
        self.network = _mlp_block(input_dim, hidden_dim1, hidden_dim2, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLP_DDS(nn.Module):
    """
    MLP con selección dinámica de características (DDS) como capa de entrada.
    El bloque clasificador es idéntico al del SimpleMLP, de modo que la única
    diferencia entre ambos modelos es la selección.
    """

    def __init__(
        self,
        input_dim: int,
        n_features_to_select: int,
        hidden_dim1: int = 50,
        hidden_dim2: int = 25,
        dropout_rate: float = 0.2,
        temperature: float = 2.0 / 3.0,
    ):
        super().__init__()
        self.dds = DynamicDataSelectionHard2(
            n_features_to_select=n_features_to_select,
            in_features=input_dim,
            temperature=temperature,
            factor_2=0.1,
            largest=True,
        )
        self.network = _mlp_block(input_dim, hidden_dim1, hidden_dim2, dropout_rate)

    def forward(self, x: torch.Tensor):
        """Devuelve (logits, penalización, máscara dura, nº de variables usadas por muestra)."""
        x_selected, hard_mask, _, penalty, n_used = self.dds(x)
        logits = self.network(x_selected)
        return logits, penalty, hard_mask, n_used


# Nombre antiguo, por compatibilidad con scripts anteriores
DynamicE2EFS_MLP = MLP_DDS
