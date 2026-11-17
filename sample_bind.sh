#!/bin/bash
# Prova diverses configuracions de col·locació (--map-by) i fixació (--bind-to)
# dels processos MPI amb un nombre fix de processos.
#
# Ús:   ./sample_bind.sh [nom_carpeta] [particio] [processos] [repeticions] [flags]
#   nom_carpeta  subcarpeta dins de logs/mpi/   (per defecte: bind-<particio>)
#   particio     cua de SLURM                   (per defecte: nodo.q)
#   processos    nombre fix de processos        (per defecte: 6)
#   repeticions  execucions per configuració    (per defecte: 3)
#   flags        flags extra de compilació      (per defecte: -DPROFILE;
#                passa "" per compilar sense instrumentació)
#
# Exemple: ./sample_bind.sh bind-nodo nodo.q 6 3
#   → logs/mpi/bind-nodo/bind_<map>-<bind>_<r>.out

PARTITION=${2:-nodo.q}
RUN_NAME=${1:-bind-${PARTITION%%.q}}
P=${3:-6}
REPS=${4:-3}
FLAGS=${5--DPROFILE}

# Parelles map-by:bind-to que es proven
CONFIGS=(
    "default:default"   # el que fa OpenMPI sense dir-li res
    "core:core"         # nuclis consecutius, cada procés fix al seu nucli
    "socket:core"       # alterna sockets, cada procés fix al seu nucli
    "core:socket"       # nuclis consecutius, cada procés pot moure's dins del socket
    "socket:socket"     # alterna sockets, cada procés pot moure's dins del socket
    "core:none"         # nuclis consecutius, sense fixar (el SO els pot moure)
)

OUT_DIR=logs/mpi/${RUN_NAME}
mkdir -p "${OUT_DIR}"   # SLURM no crea la carpeta de sortida

for cfg in "${CONFIGS[@]}"; do
    MAP=${cfg%%:*}
    BIND=${cfg##*:}
    for r in $(seq 1 "${REPS}"); do
        sbatch --partition="${PARTITION}" \
               --ntasks="${P}" \
               --job-name="bind_${MAP}-${BIND}" \
               --output="${OUT_DIR}/bind_${MAP}-${BIND}_${r}.out" \
               BIND.sub "${MAP}-${BIND}_${r}" "${P}" "${MAP}" "${BIND}" "${FLAGS}"
    done
done

echo "Enviades ${#CONFIGS[@]} configuracions x ${REPS} repeticions amb P=${P} a ${PARTITION}."
echo "Sortides a ${OUT_DIR}/"
