"""
Capa de Selección Dinámica de Características (DDS) con compuertas Concrete y Top-K.

Procedencia:
  - De la implementación original
    (DynamicDataSelectionHard2): compuertas estocásticas Concrete con temperatura 2/3,
    límites 0 y 1 (sin estiramiento), selección Top-K por muestra y factor de
    exploración `factor_2` (un 10 % de las muestras de cada lote no se poda durante
    el entrenamiento).
  - Añadido en este trabajo:
      * `gate_bias`: importancia aprendible de cada característica.
      * Selección restringida a las características ACTIVAS de cada gen (x = 1): con datos
        binarios, una característica inactiva no aporta información, así que K cuenta
        variables reales del gen.
      * Estimador Straight-Through para que el gradiente llegue a todas las compuertas.
      * Penalización sobre la suma de compuertas de las características activas que
        excede K (la del original estaba desactivada, multiplicada por 0).
"""
import torch
import torch.nn as nn

LIMIT_A = 0.0
LIMIT_B = 1.0
EPSILON = 1e-6


class DynamicDataSelectionHard2(nn.Module):
    def __init__(
        self,
        n_features_to_select: int,
        in_features: int,
        temperature: float = 2.0 / 3.0,
        factor_2: float = 0.1,
        largest: bool = True,
    ):
        super().__init__()
        self.n_features_to_select = int(n_features_to_select)
        self.in_features = int(in_features)
        self.temp = float(temperature)
        self.factor_2 = float(factor_2)
        self.largest = bool(largest)

        # Importancia aprendible de cada característica (se optimiza junto con el MLP)
        self.gate_bias = nn.Parameter(torch.zeros(self.in_features))

    @staticmethod
    def _quantile_concrete(eps: torch.Tensor, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        logistic_noise = torch.log(eps) - torch.log1p(-eps)
        return torch.sigmoid((logistic_noise + logits) / temperature)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, in_features] binario.
        Devuelve (x seleccionada, máscara dura, compuertas s, penalización, nº de variables usadas por muestra).
        """
        if x.dim() != 2:
            raise ValueError(f"DDS espera un tensor 2D [batch, features], recibido {tuple(x.shape)}")

        k = min(self.n_features_to_select, x.size(1))
        active = (x > 0).float()

        logits = x + self.gate_bias.unsqueeze(0)
        if self.training:
            eps = torch.empty_like(logits).uniform_(EPSILON, 1.0 - EPSILON)
            z = self._quantile_concrete(eps, logits, self.temp)
        else:
            z = torch.sigmoid(logits / self.temp)

        s = torch.clamp(z * (LIMIT_B - LIMIT_A) + LIMIT_A, min=0.0, max=1.0)
        s_active = s * active

        # Top-K solo entre las características activas de cada gen: las inactivas
        # reciben -1 para que nunca ganen a una activa.
        scores = torch.where(active > 0, s, torch.full_like(s, -1.0))
        _, indices = torch.topk(scores, k=k, dim=-1, largest=self.largest)
        hard_mask = torch.zeros_like(s)
        hard_mask.scatter_(-1, indices, 1.0)
        hard_mask = hard_mask * active  # si el gen tiene menos de K activas, se usan todas

        # Exploración (original): un 10 % de las muestras del lote no se podan en entrenamiento
        if self.training and self.factor_2 > 0.0:
            keep_all = torch.rand((x.size(0), 1), device=x.device) < self.factor_2
            hard_mask = torch.where(keep_all, active, hard_mask)

        # Straight-Through: hacia delante la máscara discreta; hacia atrás, gradiente continuo
        mask_st = hard_mask.detach() - s_active.detach() + s_active
        selected = x * mask_st * s

        # Penalización: cuánto excede de K la suma de compuertas de las características activas
        penalty = torch.relu(s_active.sum(dim=-1) - float(k)).mean() / float(k)

        n_used = hard_mask.sum(dim=-1)
        return selected, hard_mask, s, penalty, n_used
