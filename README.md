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

Comparamos **dos implementaciones** de KNN brute-force bajo distintas
palancas de paralelismo:

- **`sklearn-brute`** — `KNeighborsClassifier(algorithm="brute",
  metric="euclidean")`. En versiones recientes de scikit-learn este camino
  delega al kernel Cython/C++ `PairwiseDistancesArgKmin`, que paraleliza
  vía **OpenMP** internamente e ignora el argumento `n_jobs`.
- **`manual-brute`** — implementación propia: para cada query se calcula
  la distancia euclidiana contra todo `X_train` en NumPy y se vota entre
  los `k` vecinos más cercanos. La paralelización se hace explícita con
  **`joblib.Parallel`** (backend `threading` o `loky`).

Ambas implementaciones comparten un único entry-point (`knn_runner.py`),
generan el mismo schema de CSV y se barren con el mismo grid de tamaños,
de modo que las comparaciones son directas.

### Ejes del barrido

- **Muestras de entrenamiento (`N_train`)** — `--mult-train` ∈ {1, 4, 16, 64, 256}.
- **Queries de prueba (`N_test`)** — `--mult-test` ∈ {1, 16}.
- **Número de atributos (`D`)** — `--feat-mult` ∈ {1, 2, 4}, modo `mix`.
- **Threads OpenMP** — `OMP_NUM_THREADS` ∈ {1, 2, 4, 8, 16, 32} (sklearn-brute).
- **Tareas joblib** — `--jobs` ∈ {1, 2, 4, 8, 16, 32} (cross-product con OMP en sklearn-brute, sweep solo con OMP=1 en manual-brute).
- **Backend joblib** — `threading` y `loky` (manual-brute).
- `k = 5` fijo, 3 repeticiones por configuración (`--reps 3`).

Cross-product completo para sklearn-brute (6 × 6 = 36 puntos paralelos)
más sweeps de joblib en ambos backends para manual-brute → **1 440 runs**
totales sobre 30 configuraciones de tamaño.

## Mediciones y análisis

- **Tiempos de ejecución** por configuración.
- **Speedup** S(p) = T(1) / T(p).
- **Eficiencia** E(p) = S(p) / p.
- Comparación frente a la **complejidad teórica** de KNN brute-force
  (O(N·D) por consulta) vista en clase.
- **Overhead** observado al aumentar el número de hilos.
- **Escalabilidad fuerte** (N fijo, p creciente) y **escalabilidad débil**
  (trabajo por hilo aproximadamente constante).
- **Cross-product OMP × n_jobs** para sklearn-brute: heatmap 2D que
  demuestra cuál de las dos palancas realmente paraleliza.

## Estructura del repositorio

```
.
├── dataset/
│   ├── dataset.csv     # Dataset preparado (303 × 24)
│   └── dataset.md      # Documentación del preprocesamiento
├── knn_runner.py       # CLI unificado (sklearn + manual_brute)
├── knn_sweep.sh        # Job SLURM con el grid de barrido
├── results_knn.csv     # Resultados del barrido (generado por knn_sweep.sh)
└── README.md
```

## Ejecución

### Local — una configuración

```bash
# sklearn-brute con OMP=4 (paraleliza vía OpenMP, n_jobs es no-op)
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python3 knn_runner.py \
        --impl sklearn-brute --backend threading --jobs 1 \
        --mult-train 64 --mult-test 16 --feat-mult 2 \
        --reps 3 --output results_knn.csv

# manual-brute con 8 workers joblib (BLAS pineado a 1 para evitar
# oversubscription; el paralelismo viene de joblib sobre Python)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 knn_runner.py \
        --impl manual-brute --backend threading --jobs 8 \
        --mult-train 64 --mult-test 16 --feat-mult 2 \
        --reps 3 --output results_knn.csv
```

Implementaciones disponibles vía `--impl`:
`sklearn-brute`, `sklearn-kd_tree`, `sklearn-ball_tree`, `manual-brute`.

### Cluster (SLURM)

```bash
sbatch knn_sweep.sh
```

El job recorre los tres bloques de paralelismo (sklearn-brute cross-product
OMP × jobs; manual-brute threading; manual-brute loky) sobre las 30
configuraciones de tamaño, escribiendo todo a `results_knn.csv`. Walltime
fijado en `--time=02:30:00`; estimación realista ≈ 75 min.

### Schema del CSV

Cada fila representa una configuración promedio de `--reps` repeticiones:

```
implementation, omp_num_threads, n_jobs, backend, algorithm,
n_train, n_test, n_features, k,
feat_mult, feat_mode, mult_train, mult_test, jitter, reps,
fit_time_s_avg, pred_time_s_avg, accuracy_avg,
flops_distance, flops_select, flops_total,
slurm_job_id, slurm_array_task_id
```

- `implementation` ∈ {`sklearn-brute`, `sklearn-kd_tree`, `sklearn-ball_tree`, `manual-brute`}.
- `omp_num_threads` se lee de `OMP_NUM_THREADS` antes de importar NumPy.
- `algorithm` retiene el nombre que usa scikit-learn (`brute` / `kd_tree` /
  `ball_tree`) o `brute_manual` para la implementación propia.

### Análisis

Una vez generado `results_knn.csv`, el análisis se hace filtrando por
`implementation` y la palanca de paralelismo correspondiente:

- `implementation == "sklearn-brute" & n_jobs == 1`, variando
  `omp_num_threads` → strong scaling vía OpenMP (esperado: speedup hasta
  ~10× a OMP=16).
- `implementation == "sklearn-brute" & omp_num_threads == 1`, variando
  `n_jobs` → demuestra que `n_jobs` es no-op para el camino
  brute+euclidean (esperado: speedup ≈ 1).
- `implementation == "sklearn-brute"` completo → heatmap 6×6 OMP × n_jobs.
- `implementation == "manual-brute"` por backend (threading vs loky)
  → comparación de overhead de joblib.

## Autores

+ Hugo Angeles
+ Jhomar Yurivilca
+ Christian Cajusol
+ Francisco Meza

UTEC, Maestría en Ciencia de Datos e Inteligencia Artificial.
