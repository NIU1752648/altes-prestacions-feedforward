#!/bin/bash
# Envia una mostra de la versió MPI per a P = 1, 2, 4, 6, 12 processos.
#
# Ús:   ./sample_mpi.sh [nom_carpeta] [particio] [repeticions] [flags]
#   nom_carpeta  subcarpeta dins de logs/mpi/   (per defecte: data i hora)
#   particio     cua de SLURM                   (per defecte: nodo.q)
#   repeticions  execucions per a cada P        (per defecte: 1)
#   flags        flags extra de compilació      (per defecte: -DPROFILE;
#                passa "" per compilar sense instrumentació)
#
# Exemple: ./sample_mpi.sh mostra-profile nodo.q 10
#   → logs/mpi/mostra-profile/mpi_<P>p_<r>.out

RUN_NAME=${1:-$(date +%Y%m%d-%H%M%S)}
PARTITION=${2:-nodo.q}
REPS=${3:-1}
FLAGS=${4--DPROFILE}   # sense ':' perquè "" desactivi la instrumentació

OUT_DIR=logs/mpi/${RUN_NAME}
mkdir -p "${OUT_DIR}"   # SLURM no crea la carpeta de sortida

for P in 1 2 4 6 12; do
    for r in $(seq 1 "${REPS}"); do
        # El primer argument de MPI.sub fa que cada feina tingui un executable propi
        sbatch --partition="${PARTITION}" \
               --ntasks="${P}" \
               --job-name="mpi_${P}p" \
               --output="${OUT_DIR}/mpi_${P}p_${r}.out" \
               MPI.sub "${P}p_${r}" "${P}" "${FLAGS}"
    done
done

echo "Feines enviades a ${PARTITION} (flags: '${FLAGS}'). Sortides a ${OUT_DIR}/"
