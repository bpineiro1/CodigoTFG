# Priorización genética mediante selección dinámica de características — versión 1

Comparación de un MLP base frente a un MLP con capa de selección dinámica de características (DDS)
en los conjuntos GO y PathDIP (genes relacionados con la restricción calórica).

## Estructura

```
train_mlp.py          MLP base: validación cruzada anidada repetida
train_dds.py          MLP + DDS: mismo protocolo, mismas particiones y candidatos
compare_models.py     Tabla comparativa y tests estadísticos (pareado y corregido)
check_overfitting.py  Curvas de entrenamiento/validación para revisar sobreajuste
Code/
  model.py            SimpleMLP y MLP_DDS
  dds.py              Capa DDS (compuertas Concrete + Top-K entre variables activas)
  metrics.py          Métricas, umbral por G-Mean y predicción por fracción de positivos
  plots.py            Gráficas y tabla resumen
Data/                 GO.csv y PathDip.csv (no incluidos)
```

## Protocolo

- Validación cruzada anidada 10 (externa) × 3 (interna), repetida con 10 semillas:
  14, 33, 39, 42, 727, 1312, 1337, 56709, 177013, 241543903.
- Búsqueda aleatoria de 20 candidatos por fold; métrica interna AUC-PR.
- Hiperparámetros elegidos en el bucle interno: épocas, batch, learning rate, peso de clase,
  weight decay, dropout y, en el DDS, K y penalización.
- Umbral por G-Mean elegido en el bucle interno y trasladado al test como fracción de genes positivos.
- Resultados como media ± desviación típica entre las 10 repeticiones.

## Ejecución

```
python train_mlp.py | tee log_mlp.txt
python train_dds.py | tee log_dds.txt
python compare_models.py
```

Opciones: `--datasets GO PathDip` y `--repeats N`. Cada semilla terminada se guarda en
`results/reps/`, así que si se interrumpe, al relanzar continúa desde la última completada.
Los resultados se guardan en `results/`.
