#!/bin/bash
#SBATCH --job-name=knn_sweep
#SBATCH --output=knn_sweep_out_%j.txt
#SBATCH --error=knn_sweep_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --partition=standard

echo "Inicio: $(date)"
echo "CPUs asignados: ${SLURM_CPUS_PER_TASK}"

CSV="results_knn.csv"
rm -f "${CSV}"   # batch limpio

ml load python3
source ~/venv/bin/activate

# Sanity: el runner debe leer OMP del env (no setdefault)
echo "--- sanity check: knn_runner.py lee OMP del env ---"
OMP_NUM_THREADS=7 python3 -c "
import os
print('OMP visible antes de import:', os.environ.get('OMP_NUM_THREADS','<unset>'))
"
echo "---"

# Grids
OMP_GRID=(1 2 4 8 16 32)
JOBS_GRID=(1 2 4 8 16 32)
GRID_MT=(1 4 16 64 256)        # mult_train
GRID_MTE=(1 16)                # mult_test
GRID_FM=(1 2 4)                # feat_mult

# Total esperado:
#   Bloque A: |OMP|*|JOBS|*|MT|*|MTE|*|FM| = 6*6*5*2*3 = 1080
#   Bloque B: |JOBS|*|MT|*|MTE|*|FM|       = 6*5*2*3   = 180
#   Bloque C: idem                                     = 180
#   Total = 1440
total=$(( ${#OMP_GRID[@]}*${#JOBS_GRID[@]}*${#GRID_MT[@]}*${#GRID_MTE[@]}*${#GRID_FM[@]} \
        + 2*${#JOBS_GRID[@]}*${#GRID_MT[@]}*${#GRID_MTE[@]}*${#GRID_FM[@]} ))
echo "Total runs esperados: ${total}"

run_one() {
    local impl="$1" omp="$2" jobs="$3" backend="$4" mt="$5" mte="$6" fm="$7"
    # Skip combinaciones que excedan los cores asignados
    if [ "${omp}" -gt "${SLURM_CPUS_PER_TASK}" ]; then return; fi
    if [ "${jobs}" -gt "${SLURM_CPUS_PER_TASK}" ]; then return; fi
    OMP_NUM_THREADS=${omp} OPENBLAS_NUM_THREADS=${omp} MKL_NUM_THREADS=${omp} \
    VECLIB_MAXIMUM_THREADS=${omp} NUMEXPR_NUM_THREADS=${omp} \
        python3 knn_runner.py \
            --impl "${impl}" --backend "${backend}" --jobs "${jobs}" --k 5 \
            --mult-train "${mt}" --mult-test "${mte}" --feat-mult "${fm}" \
            --feat-mode mix --jitter 0.1 --reps 3 \
            --output "${CSV}"
}

counter=0

for mt in "${GRID_MT[@]}"; do
 for mte in "${GRID_MTE[@]}"; do
  for fm in "${GRID_FM[@]}"; do

    # ----- Bloque A: sklearn-brute cross-product OMP x jobs -----
    for omp in "${OMP_GRID[@]}"; do
      for j in "${JOBS_GRID[@]}"; do
        counter=$((counter+1))
        echo "[${counter}/${total}] A sklearn-brute omp=${omp} jobs=${j} mt=${mt} mte=${mte} fm=${fm}"
        run_one sklearn-brute "${omp}" "${j}" threading "${mt}" "${mte}" "${fm}"
      done
    done

    # ----- Bloque B: manual-brute threading, OMP=1, sweep jobs -----
    for j in "${JOBS_GRID[@]}"; do
      counter=$((counter+1))
      echo "[${counter}/${total}] B manual-brute(threading) omp=1 jobs=${j} mt=${mt} mte=${mte} fm=${fm}"
      run_one manual-brute 1 "${j}" threading "${mt}" "${mte}" "${fm}"
    done

    # ----- Bloque C: manual-brute loky, OMP=1, sweep jobs -----
    for j in "${JOBS_GRID[@]}"; do
      counter=$((counter+1))
      echo "[${counter}/${total}] C manual-brute(loky) omp=1 jobs=${j} mt=${mt} mte=${mte} fm=${fm}"
      run_one manual-brute 1 "${j}" loky "${mt}" "${mte}" "${fm}"
    done

  done
 done
done

echo "Fin: $(date)"
echo "CSV final: ${CSV} ($(wc -l < ${CSV}) lineas incluyendo header)"
