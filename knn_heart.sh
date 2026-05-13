#!/bin/bash
#SBATCH --job-name=knn_heart_sklearn
#SBATCH --output=knn_heart_out_%j.txt
#SBATCH --error=knn_heart_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=standard

echo "Inicio del experimento: $(date)"
echo "CPUs asignados por SLURM: ${SLURM_CPUS_PER_TASK}"

CSV_FILE="results_knn_heart.csv"

# Grid de barrido (todos los combinaciones se ejecutan dentro de la misma
# asignacion SLURM para que los tiempos sean comparables entre si).
THREADS_GRID=(1 2 4 8 16 32)
MULT_TRAIN_GRID=(32 64 128 256 512 1024)
FEAT_MULT_GRID=(1 2 4 8)

for n_jobs in "${THREADS_GRID[@]}"; do
    # No pedir mas hilos que CPUs asignados por SLURM.
    if [ "${n_jobs}" -gt "${SLURM_CPUS_PER_TASK}" ]; then
        echo "Skip n_jobs=${n_jobs} (> SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK})"
        continue
    fi
    for mult_train in "${MULT_TRAIN_GRID[@]}"; do
        for feat_mult in "${FEAT_MULT_GRID[@]}"; do
            echo "==> n_jobs=${n_jobs} mult_train=${mult_train} feat_mult=${feat_mult}"
            python3 knn_heart_sklearn_scale.py \
                --k 5 \
                --jobs ${n_jobs} \
                --mult-train ${mult_train} \
                --mult-test 2 \
                --feat-mult ${feat_mult} \
                --feat-mode mix \
                --backend threading \
                --algorithm brute \
                --jitter 0.1 \
                --reps 3 \
                --output ${CSV_FILE}
        done
    done
done

echo "Fin del experimento: $(date)"
