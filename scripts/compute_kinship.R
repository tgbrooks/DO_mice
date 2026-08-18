library(tidyverse)
library(qtl2)

load("geno/genoprobs.RData")

K   <- calc_kinship(genoprobs, type = "loco")

for (chromosome in names(K)) {
    write_tsv(K[[chromosome]] |> as.data.frame() |> rownames_to_column("mouse_id"), paste0("geno/kinship/", chromosome, ".txt"))
}
