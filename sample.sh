
# OpenMP -fno-tree-vectorize
for i in {1,2,4,6,12}; do
    for j in {1..10}; do
        sbatch --job-name=omp_${i}threads \
        --output=logs/openmp/actually-final/omp_${i}threads_${j}.out \
        OMP.sub $j $i
    done
done

# OMP + SIMD -DUSE_SIMD
for i in {1,2,4,6,12}; do
    for j in {1..10}; do
        sbatch --job-name=simd_${i}threads \
        --output=logs/openmp/actually-final/simd_${i}threads_${j}.out \
        SIMD.sub $j $i
    done
done
