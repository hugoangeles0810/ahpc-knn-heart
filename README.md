# Tarea 2 — Escalabilidad de KNN con datos médicos

Trabajo del curso **Applied High Performance Computing (AHPC)** de la
Maestría en **Ciencia de Datos e Inteligencia Artificial** de **UTEC**.

## Tema

**Predicción de enfermedad cardíaca mediante K-Nearest Neighbors.**

Una institución de salud desea construir un modelo de IA que ayude a
clasificar si un paciente presenta indicios de enfermedad cardíaca a partir
de variables clínicas como edad, presión arterial, colesterol, frecuencia
cardíaca máxima y tipo de dolor torácico.

El **enfoque no es la precisión del modelo**, sino el **costo computacional
y el análisis de escalabilidad** al variar el tamaño del problema y el
paralelismo.

## Dataset

[Heart Disease Dataset (UCI Repository)](https://archive.ics.uci.edu/dataset/45/heart+disease)
— subconjunto Cleveland: 303 registros, 13 variables clínicas y una
variable objetivo de clasificación.

El detalle del preprocesamiento (imputación, codificación one-hot, columnas
finales) está en [`dataset/dataset.md`](dataset/dataset.md). El artefacto
listo para consumir es [`dataset/dataset.csv`](dataset/dataset.csv).

## Metodología

- Implementación en **Python** con `scikit-learn` (`KNeighborsClassifier`,
  backend `threading`, algoritmo `brute`).
- División **70% entrenamiento / 30% prueba**.
- Barrido sobre cuatro ejes de escalabilidad:
  - **Muestras de entrenamiento (`N_train`)** — `--mult-train` ∈ {1, 4, 16, 64, 256, 1024}.
  - **Queries de prueba (`N_test`)** — `--mult-test` ∈ {1, 16, 128}.
  - **Número de atributos (`D`)** — `--feat-mult` ∈ {1, 2, 4, 8}, modo `mix`.
  - **Número de procesos/hilos** — `--jobs` ∈ {1, 2, 4, 8, 16, 32}.
- Repeticiones por configuración (`--reps`) para reducir ruido de medición.

## Mediciones y análisis

- **Tiempos de ejecución** por configuración.
- **Speedup** S(p) = T(1) / T(p).
- **Eficiencia** E(p) = S(p) / p.
- Comparación frente a la **complejidad teórica** de KNN brute-force
  (O(N·D) por consulta) vista en clase.
- **Overhead** observado al aumentar el número de hilos.
- **Escalabilidad fuerte** (N fijo, p creciente) y **escalabilidad débil**
  (trabajo por hilo aproximadamente constante).

## Estructura del repositorio

```
.
├── dataset/
│   ├── dataset.csv             # Dataset preparado (303 × 24)
│   └── dataset.md              # Documentación del preprocesamiento
├── knn_heart_sklearn_scale.py  # Script de benchmark
├── knn_heart.sh                # Job SLURM con grid de barrido
├── knn_heart_analisis.ipynb    # Notebook de visualizaciones y análisis
├── results_knn_heart.csv       # Resultados del barrido (generado)
└── README.md
```

## Ejecución

### Local

```bash
python3 knn_heart_sklearn_scale.py \
    --k 5 --jobs 4 \
    --mult-train 64 --mult-test 16 \
    --feat-mult 2 --feat-mode mix \
    --backend threading --algorithm brute \
    --reps 3 --output results_knn_heart.csv
```

### Cluster (SLURM)

```bash
sbatch knn_heart.sh
```

El job barre el producto cartesiano de `THREADS_GRID × MULT_TRAIN_GRID ×
MULT_TEST_GRID × FEAT_MULT_GRID` (≈432 configuraciones) y acumula los
tiempos en `results_knn_heart.csv`. El walltime SLURM está fijado en
`--time=04:00:00`.

### Visualización de resultados

Una vez generado `results_knn_heart.csv`, abrir
[`knn_heart_analisis.ipynb`](knn_heart_analisis.ipynb) en Jupyter / VS Code
y ejecutar todas las celdas. El notebook detecta automáticamente los ejes
del barrido y produce las gráficas de tiempo, speedup, eficiencia,
overhead, escalabilidad fuerte/débil y comparación con la complejidad
teórica.

## Autores

+ Hugo Angeles
+ Jhomar Yurivilca
+ Christian Cajusol
+ Francisco Meza

UTEC, Maestría en Ciencia de Datos e Inteligencia Artificial.