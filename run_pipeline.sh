#!/usr/bin/env bash
# Submit the DO mouse pipeline to LSF. Adjust the project, queue, and bind path
# below for your cluster. Extra arguments are passed to Snakemake, e.g.
#   ./run_pipeline.sh -n
#   ./run_pipeline.sh genotyped_mice

# disable FIPS
echo 0 > "$PWD/fips_off"
APPTAINER_TMPDIR=/scratch/tmp bsub -e logs/snakemake.err \
    -o logs/snakemake.out \
    uv run snakemake \
    -c 100 \
    -j 100 \
    --executor lsf \
    --default-resources lsf_project=DO_mice lsf_queue=rhel9 mem_mb=4000 \
    --resources sra_downloads=5 \
    --use-apptainer \
    --apptainer-args "--bind \"$PWD/fips_off\":/proc/sys/crypto/fips_enabled" \
    "$@"
