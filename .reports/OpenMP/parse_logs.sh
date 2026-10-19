#!/usr/bin/env bash
# Extreu temps i encerts dels logs de Slurm i genera dos CSV:
#   raw.csv      una fila per execució (per reutilitzar)
#   summary.csv  una fila per (versió, fils), sense capçalera, en l'ordre de
#                la taula de main.typ: Versió, Threads, Execucions, mu, min t,
#                max t, sigma, S(n), E(n), min C, max C
#
# S(n) = T_seq / mu(n), amb T_seq el temps seqüencial original.
# E(n) = T_1 / (n * mu(n)), amb T_1 la mitjana d'1 fil del nostre codi (mateixa versió).
#
# Ús: ./parse_logs.sh [dir_logs] [log_seqüencial] [dir_sortida]

set -euo pipefail

LOGS="${1:-../logs/openmp/actually-final}"
SEQ="${2:-../logs/openmp/sequential/01.out}"
OUT="${3:-.}"

# Temps de referència seqüencial: "Goodbye! (40.288796 sec)"
T1=$(sed -nE 's/.*Goodbye! \(([0-9.]+) sec\).*/\1/p' "$SEQ" | head -n1)
[ -n "$T1" ] || { echo "No s'ha trobat el temps a $SEQ" >&2; exit 1; }

RAW="$OUT/raw.csv"
SUMMARY="$OUT/summary.csv"

echo "version,threads,run,time_s,hits,total,final_loss,file" > "$RAW"

for f in "$LOGS"/*_*threads_*.out; do
    [ -e "$f" ] || { echo "Cap log a $LOGS" >&2; exit 1; }
    name=$(basename "$f" .out)                 # p.ex. simd_12threads_3
    version=$(echo "$name" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')
    threads=$(echo "$name" | sed -E 's/^[a-z]+_([0-9]+)threads_.*/\1/')
    run=$(echo "$name" | sed -E 's/.*_([0-9]+)$/\1/')

    time_s=$(sed -nE 's/.*Goodbye! \(([0-9.]+) sec\).*/\1/p' "$f" | tail -n1)
    hits=$(sed -nE 's/.*Total encerts = ([0-9]+) \(\/([0-9]+)\).*/\1/p' "$f" | tail -n1)
    total=$(sed -nE 's/.*Total encerts = ([0-9]+) \(\/([0-9]+)\).*/\2/p' "$f" | tail -n1)
    loss=$(sed -nE 's/^Epoch [0-9]+ Loss: ([0-9.]+).*/\1/p' "$f" | tail -n1)

    if [ -z "$time_s" ] || [ -z "$hits" ]; then
        echo "AVÍS: dades incompletes a $f (omès)" >&2
        continue
    fi
    echo "$version,$threads,$run,$time_s,$hits,$total,$loss,$name.out" >> "$RAW"
done

# Resum per (versió, fils). sigma = desviació estàndard mostral (n-1).
# Ordena per versió (OMP abans que SIMD) i fils numèricament.
# T_1 del nostre codi per versió (mitjana de les execucions d'1 fil)
T1_OMP=$(awk -F, '$1=="OMP" && $2==1 {s+=$4; n++} END {if (n) printf "%.6f", s/n}' "$RAW")
T1_SIMD=$(awk -F, '$1=="SIMD" && $2==1 {s+=$4; n++} END {if (n) printf "%.6f", s/n}' "$RAW")
[ -n "$T1_OMP" ] && [ -n "$T1_SIMD" ] || { echo "Falten execucions d'1 fil a $RAW" >&2; exit 1; }

tail -n +2 "$RAW" | sort -t, -k1,1 -k2,2n | awk -F, -v t1="$T1" -v o1="$T1_OMP" -v s1="$T1_SIMD" '
function flush(   mu, var, s, e, sd) {
    if (n == 0) return
    mu = sum / n
    var = (n > 1) ? (sumsq - n * mu * mu) / (n - 1) : 0
    if (var < 0) var = 0
    sd = sqrt(var)
    s = t1 / mu
    e = ((ver == "OMP") ? o1 : s1) / (mu * thr)
    printf "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.3f,%d,%d\n", \
        ver, thr, n, mu, mn, mx, sd, s, e, cmin, cmax
}
{
    key = $1 "," $2
    if (key != last) {
        flush()
        last = key; ver = $1; thr = $2
        n = 0; sum = 0; sumsq = 0; mn = 1e18; mx = -1e18
        cmin = 1e18; cmax = -1e18
    }
    t = $4 + 0; c = $5 + 0
    n++; sum += t; sumsq += t * t
    if (t < mn) mn = t; if (t > mx) mx = t
    if (c < cmin) cmin = c; if (c > cmax) cmax = c
}
END { flush() }
' > "$SUMMARY"

echo "T_seq original = $T1 s | T_1 nostre: OMP $T1_OMP s, SIMD $T1_SIMD s"
echo "Escrit: $RAW ($(($(wc -l < "$RAW") - 1)) execucions)"
echo "Escrit: $SUMMARY"
