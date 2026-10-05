# Compute tximport-style size factors with DESeq2 median-of-ratios method.
# Specifically, these normalize the count / effective_length values
# with effective-length values estimated (per gene and per sample) by Salmon
# Resulting normalization factors are then size_factor * effective_length

library(DESeq2)
library(tidyverse)
library(arrow)

args <- commandArgs(trailingOnly = TRUE)
count_file <- args[1]
effective_lengths_file <- args[2]
outfile <- args[3]
out_normfile <- args[4]

counts <- read_parquet(count_file)
counts_mat <- pivot_wider(
    counts,
    id_cols = "gene_id",
    names_from = "mouse_id",
    values_from = "total_reads",
) |>
    column_to_rownames("gene_id") |>
    as.matrix()
storage.mode(counts_mat) <- "integer"

effective_lengths <- read_parquet(effective_lengths_file)
effective_lengths_mat <- effective_lengths |>
    pivot_wider(
        id_cols = all_of("gene_id"),
        names_from = "mouse_id",
        values_from = "effective_length",
    ) |>
    column_to_rownames("gene_id") |>
    as.matrix()
effective_lengths_mat <- effective_lengths_mat[rownames(counts_mat),][,colnames(counts_mat)]

dummy <- tibble(
    mouse_id = colnames(counts_mat),
    dummy = 1,
)

dds <- DESeqDataSetFromMatrix(
    countData = counts_mat,
    colData = dummy,
    design = ~ 1,
)

norm_mat <- effective_lengths_mat / exp(rowMeans(log(effective_lengths_mat))) # center rows on 1
res <- estimateSizeFactors(dds, normMatrix = norm_mat)
nf <- normalizationFactors(res)

# Extract per-sample size factors
size_factors <- (nf / norm_mat)[1,] # all rows are the same, use the first one
tibble(
    mouse_id = names(size_factors),
    size_factor = size_factors,
) |> write_tsv(outfile)

# Per-sample, per-gene normalization factors
nf |> rownames_to_column("gene_id") |>
    pivot_longer(
        cols = all_of(colnames(nf)),
        names_to = "mouse_id",
        values_to = "normalization_factor",
    ) |>
    write_tsv(out_normfile)
