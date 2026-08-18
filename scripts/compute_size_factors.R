library(DESeq2)
library(tidyverse)
library(arrow)

count_file <- "results/Adipose/Adipose.diploid.genes.founder_expected_read_counts.parquet"
outfile <- "results/Adipose/size_factors.txt"
count_file <- snakemake@input$counts
outfile <- snakemake@output$outfile

counts <- read_parquet(count_file)
counts_mat <- pivot_wider(
    counts,
    id_cols = "gene_id",
    names_from = "mouse_id",
    values_from = "total",
) |>
    column_to_rownames("gene_id") |>
    as.matrix()
storage.mode(counts_mat) <- "integer"

dummy <- tibble(
    mouse_id = colnames(counts_mat),
    dummy = 1,
)

dds <- DESeqDataSetFromMatrix(
    countData = counts_mat,
    colData = dummy,
    design = ~ 1,
)

size_factors <- estimateSizeFactors(dds)
tibble(
    mouse_id = names(colData(size_factors)$sizeFactor),
    size_factor = colData(size_factors)$sizeFactor,
) |> write_tsv(outfile)
