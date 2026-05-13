# Preparación de Datos — `dataset.csv`

Este documento describe la canalización de imputación y codificación aplicada
al subconjunto Cleveland del dataset UCI Heart Disease para producir
`dataset.csv`, la versión destinada a alimentar un modelo basado en distancias
(por ejemplo, KNN).

> **Propósito de este dataset.** Se utiliza como entrada para **benchmarks de
> paralelización** de una implementación de KNN — variando el número de hilos,
> el tamaño del dataset (`N`), la cantidad de características (`D`) y `k`. **La
> exactitud predictiva está fuera del alcance.** Esto cambia qué pasos de
> preprocesamiento son requeridos frente a opcionales; ver §5.

## 1. Fuente

- **Origen:** subconjunto Cleveland procesado del dataset UCI Heart Disease.
- **Filas:** 303 pacientes
- **Columnas originales:** 14 (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`)

## 2. Inventario de columnas

| Columna    | Rol           | Tipo               | Valores / Significado                                       |
|------------|---------------|--------------------|-------------------------------------------------------------|
| `age`      | característica| numérico           | años                                                        |
| `sex`      | característica| binario            | 1 = masculino, 0 = femenino                                 |
| `cp`       | característica| **nominal**        | 1 angina típica, 2 atípica, 3 no anginosa, 4 asintomática   |
| `trestbps` | característica| numérico           | presión arterial en reposo (mm Hg)                          |
| `chol`     | característica| numérico           | colesterol sérico (mg/dl)                                   |
| `fbs`      | característica| binario            | azúcar en sangre en ayunas > 120 mg/dl                      |
| `restecg`  | característica| **nominal**        | 0 normal, 1 anormalidad ST-T, 2 hipertrofia VI              |
| `thalach`  | característica| numérico           | frecuencia cardíaca máxima alcanzada                        |
| `exang`    | característica| binario            | angina inducida por ejercicio                               |
| `oldpeak`  | característica| numérico           | depresión ST vs reposo                                      |
| `slope`    | característica| **nominal**        | 1 ascendente, 2 plana, 3 descendente                        |
| `ca`       | característica| ordinal numérico   | número de vasos mayores coloreados (0–3)                    |
| `thal`     | característica| **nominal**        | 3 normal, 6 defecto fijo, 7 defecto reversible              |
| `num`      | objetivo      | ordinal            | 0 sin enfermedad, 1–4 severidad creciente                   |

Las columnas marcadas como **nominal** no están ordenadas y no deben pasarse a
una métrica de distancia como enteros crudos — se codifican con one-hot a
continuación.

## 3. Auditoría de valores faltantes

Las entradas faltantes aparecen como celdas vacías (originalmente `?` en el
archivo UCI sin procesar).

| Columna | Filas faltantes | Estrategia                              |
|---------|-----------------|-----------------------------------------|
| `ca`    | 4               | **Imputación por mediana** (ordinal numérico) |
| `thal`  | 2               | **Imputación por moda** (nominal)              |
| otros   | 0               | —                                       |

Total afectado: 6 de 303 filas (~2%). También es viable descartar las filas
aquí, pero la imputación preserva la muestra completa y coincide con lo que
nos veremos obligados a hacer para los otros subconjuntos UCI (Hungarian /
Switzerland / VA), así que la adoptamos como estrategia por defecto también
en Cleveland por consistencia.

### Por qué estas estrategias

- **`ca` → mediana.** El rango es pequeño (0–3) y la distribución está
  sesgada a la derecha (la mayoría de los pacientes tienen 0 vasos
  coloreados). La media se vería arrastrada por unos pocos `3` y produciría
  un valor no entero (~0.7) sin significado clínico. La mediana produce un
  entero válido dentro del rango.
- **`thal` → moda.** `thal` es categórica (3 / 6 / 7), por lo que ni la media
  ni la mediana aplican. La moda (`3` = normal) es el prior más seguro para
  un código diagnóstico faltante.

Las estadísticas de imputación se calculan **sobre el archivo Cleveland
completo** en este paso de preparación. Cuando dividas en entrenamiento/prueba
para modelado, deberías reajustar el imputador **solo sobre el conjunto de
entrenamiento** para evitar fuga de información — este script existe para
producir un artefacto estático limpio, no para reemplazar un `Pipeline` de
sklearn adecuado en tiempo de entrenamiento.

## 4. Codificación

### Codificación one-hot nominal

Las siguientes columnas se expanden con `pd.get_dummies(..., dtype=int)`:

| Original  | Columnas resultantes                       |
|-----------|--------------------------------------------|
| `cp`      | `cp_1, cp_2, cp_3, cp_4`                   |
| `restecg` | `restecg_0, restecg_1, restecg_2`          |
| `slope`   | `slope_1, slope_2, slope_3`                |
| `thal`    | `thal_3, thal_6, thal_7`                   |

No se elimina ninguna columna dummy (`drop_first=False`). KNN no sufre de
multicolinealidad, y mantener todas las dummies hace que la geometría sea
simétrica: cada categoría contribuye con la misma distancia de Hamming (1)
cuando dos pacientes difieren.

### Mantenidas tal cual

- `sex`, `fbs`, `exang` — ya son 0/1.
- `ca` — tratada como ordinal numérico (0–3) tras la imputación.
- `age`, `trestbps`, `chol`, `thalach`, `oldpeak` — numéricos continuos.

### Objetivo

- `num` se preserva sin cambios (0–4) para experimentos multiclase.
- Se añade una columna adicional `target`: `1 si num > 0, si no 0`, la
  etiqueta binaria estándar de "enfermedad presente" usada en casi todos los
  experimentos publicados sobre Cleveland.

## 5. Lo que *no* se hace aquí — y por qué no importa para este caso de uso

### Omitido: escalado de características

Para un modelo KNN **predictivo**, el escalado (`StandardScaler` /
`MinMaxScaler`) sería obligatorio — `chol` (~200) dominaría a `oldpeak` (~1)
en la distancia euclidiana, y la exactitud colapsaría.

Para un **benchmark de paralelización**, el escalado **no es necesario**:

- El tiempo de reloj de una consulta KNN por fuerza bruta depende de `N`
  (muestras), `D` (características), `k`, y el número de hilos — no de las
  magnitudes de los números.
- Calcular `(a − b)²` cuesta la misma cantidad de FLOPs ya sea que `a − b`
  sea `0.01` o `438`.
- La huella en caché, el ancho de banda de memoria y las curvas de speedup
  son idénticas antes y después del escalado.

Por tanto, el CSV se entrega **sin escalar** a propósito. Si alguna variante
del benchmark también quisiera medir la calidad predictiva, agrega un paso
`StandardScaler` en ese punto — no aquí.

### Omitido: división entrenamiento/prueba, estratificación

Estas pertenecen a un flujo de evaluación predictiva. Para experimentos de
tiempo típicamente alimentas el dataset completo (o una versión replicada/
seccionada) a la rutina KNN y mides el rendimiento.

### Útil para benchmarking: escalar `N` y `D`

Para producir curvas significativas de escalado fuerte/débil generalmente
necesitas entradas **más grandes** que 303 × 23. Trucos comunes, todos
ortogonales al preprocesamiento anterior:

- **Aumentar `N`** replicando filas del CSV preparado.
- **Aumentar `D`** añadiendo columnas sintéticas (por ejemplo, ruido
  gaussiano independiente) cuando quieras estudiar el costo de la
  dimensionalidad.
- **Variar `k`** en tiempo de consulta para ver el costo de heap/orden.

Ninguno de estos requiere regenerar `dataset.csv`.

## 6. Salida

- **Archivo:** `dataset/dataset.csv`
- **Forma:** 303 × 24
- **Columnas finales (en orden):**
  `age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca, num,
  cp_1, cp_2, cp_3, cp_4,
  restecg_0, restecg_1, restecg_2,
  slope_1, slope_2, slope_3,
  thal_3, thal_6, thal_7,
  target`
- **Valores faltantes:** 0
