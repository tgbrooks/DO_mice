library(tidyverse)
library(glmmTMB)
library(arrow)


count_file <- "results/Adipose/Adipose.diploid.genes.founder_expected_read_counts.parquet"
chromosome <- "5"
tissue <- "Adipose"
MIN_MEDIAN_COUNTS <- 50
size_factors_file <- "results/Adipose/size_factors.txt"
annot_file <- "processed/gene_annot.txt"
outfile <- "temp.txt"
kinship_file <- paste0("geno/kinship/", chromosome, ".txt")

count_file <- snakemake@input$counts
size_factors_file <- snakemake@input$size_factors
annot_file <- snakemake@input$annot
kinship_file <- snakemake@input$kinship
chromosome <- snakemake@wildcards$chromosome
tissue <- snakemake@wildcards$tissue
MIN_MEDIAN_COUNTS <- snakemake@params$min_median_counts
outfile <- snakemake@output$outfile


annot <- read_tsv(annot_file)
gene_ids <- (annot |> filter(chrom == chromosome))$gene_id

counts <- read_parquet(count_file)
mouse_ids <- unique(counts$mouse_id)

# Kinship matrix
K   <- read_tsv(kinship_file)
K2  <- 2 * as.matrix(column_to_rownames(K, "mouse_id"))
K2 <- K2[mouse_ids, mouse_ids]

# Allele-specific counts
temp <- list()
for (mouse_id in mouse_ids) {
    temp[[length(temp)+1]] <- (
        read_parquet(
            paste0("processed/", tissue, "/gbrs_allele_unique_reads/", mouse_id, ".allele_unique_reads.parquet")
        ) |> mutate(mouse_id=mouse_id)
    )
}
allele_unique <- bind_rows(temp)

size_factors <- read_tsv(size_factors_file)
phenotypes <- read_csv("phenotypes.csv.gz")
genotypes <- read_parquet("results/genotypes.parquet") |>
    mutate(
        A = str_count(genotype, "A") ,
        B = str_count(genotype, "B") ,
        C = str_count(genotype, "C") ,
        D = str_count(genotype, "D") ,
        E = str_count(genotype, "E") ,
        F = str_count(genotype, "F") ,
        G = str_count(genotype, "G") ,
        H = str_count(genotype, "H") ,
    )

fit_model <- function(au, dilution_factor = 1, downsample_factor = 1) {
    # Fit our full buffering model
    #
    # Setting dilution_factor < 1 increases our estimation error of the cis_effect
    # from allele-specific expression by downsampling reads. This allows us to
    # examine the effect of noise in cis_effect on the final buffering value
    # This can also be done by subsetting samples au with downsample_factor < 1
    #
    # First we do a allele-specific binomial model
    # We do not include sex, DOwave, or kinship since these are always the same for both haplotypes
    # of a sample and the signs are impossible to compare across samples: hap1 and hap2 may refer to different
    # haplotypes in different samples.
    #
    # Note that if the expression of each allele follows:
    # y_i ~ NBinom(exp(X_i Beta), alpha)
    # where X_i is the vector of haplotype counts
    # then
    # p := E[y1 / (y_1 + y_2)]
    #   ≈ E[y1] / (E[y_1] + E[y_2])
    #   = exp(X_1 Beta) / (exp(X_1 Beta) + exp(X_2 beta))
    #   = exp((X_1 - X_2) Beta) / (exp((X_1 - X_2) Beta) + 1)
    #   = logit^-1( (X_1 - X_2) Beta )
    # So we use a binomial GLM with logistic link and estimator X_1 - X_2
    # where X_1 is the vector of haplotype counts for first haplotype
    # and X_2 is that of the second haplotype.
    # The approximation in the expectation step seems to be quite accurate.


    # au: allele_unique filtered to one gene

    au2 <- au |>
        filter(
            !is_homozygous,
        ) |>
        # Dilute if specified (does nothing for dilution_factor = 1)
        mutate(
               haplotype_1_unique = rbinom(length(haplotype_1_unique), haplotype_1_unique, dilution_factor),
               haplotype_2_unique = rbinom(length(haplotype_2_unique), haplotype_2_unique, dilution_factor),
        ) |>
        filter((haplotype_1_unique > 0) | (haplotype_2_unique > 0))

    if (downsample_factor < 1) {
        au2 <- au2 |> slice_sample(prop=downsample_factor)
    }

    # Fit a logit binomial model for the two haplotype unique counts
    # this estimates the cis effects in a manner that is independent of buffering
    # since buffering affects both haplotypes
    res_binom <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ effect_A + effect_B + effect_C + effect_D + effect_E + effect_F + effect_G,  # H is reference
        family = binomial,
        data = au2,
    )
    res_binom_restricted <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ 1,
        family = binomial,
        data = au2,
    )
    anova_binom <- anova(res_binom_restricted, res_binom)

    effect_A = fixef(res_binom)$cond['effect_A']
    effect_B = fixef(res_binom)$cond['effect_B']
    effect_C = fixef(res_binom)$cond['effect_C']
    effect_D = fixef(res_binom)$cond['effect_D']
    effect_E = fixef(res_binom)$cond['effect_E']
    effect_F = fixef(res_binom)$cond['effect_F']
    effect_G = fixef(res_binom)$cond['effect_G']

    ## Now we fit the buffering model, using the binomial-fit parameters
    ## We assume that tot = y_1  + y_2 is buffered by some function of itself
    ## log(tot) = cis_effect + b log(tot)
    ## log(tot) = k * cis_effect
    ## where k = 1/(1-b)
    ## and b gives the extent of buffering
    ## and we know the total cis effect from the binomial model above.
    ## NOTE: one limitation of this estimate is that there is noise in the exogenous variable
    ## cis_effect, which can dilute the slope and therefore make it look like buffering.
    buffering_data <- au |>
        mutate(
            cis_effect = effect_A*A + effect_B*B + effect_C*C + effect_D*D + effect_E*E + effect_F*F + effect_G*G # H is reference
        )

    # NOTE: we don't include generation since it's not in the available phenotypes
    res_buffering <- glmmTMB(
        total ~ offset(log(size_factor)) + sex + DOwave +
                cis_effect +
                propto(0 + mouse_id | dummy, K2),
        family = nbinom2,
        data = buffering_data,
    )

    buffering_factor <- 1 - 1/fixef(res_buffering)$cond['cis_effect']
    results <- tibble(
        buffering_factor = buffering_factor,
        dilution_factor = dilution_factor,
        downsample_factor = downsample_factor,
        n_samples_binom = nrow(au2),
        n_samples_buffering = nrow(buffering_data),
        converged_binom = res_binom$fit$converge,
        converged_buffering = res_buffering$fit$converge,
        effect_A = effect_A,
        effect_B = effect_B,
        effect_C = effect_C,
        effect_D = effect_D,
        effect_E = effect_E,
        effect_F = effect_F,
        effect_G = effect_G,
        effect_H = 0, # Reference, 0 by definition
        anova_binom_p = anova_binom$`Pr(>Chisq)`[2],
        anova_binom_chisq = anova_binom$Chisq[2],
    )
    return(results)
}


###### GENOTYPE MODEL
temp <- list()
for (gene in gene_ids) {
    if (!(gene %in% counts$gene_id)) {
        message("Skipping ", gene, " not in counts")
        next
    }

    data <- counts |> 
        filter(gene_id == gene)
    median_counts <- data$total |> median()
    if (median_counts < MIN_MEDIAN_COUNTS) {
        message("Skipping ", gene, " too low expressed")
        next
    }

    message("Running ", gene)

    # Data for the buffering model
    basic_data <- data |>
        select(gene_id, mouse_id, total) |>
        left_join(
            phenotypes,
            join_by(mouse_id == mouse.id)
        ) |>
        left_join(
            genotypes |>
                filter(gene_id == gene),
            join_by(
                mouse_id == mouse_id,
                gene_id == gene_id,
            )
        ) |>
        left_join(size_factors, "mouse_id") |>
        mutate(
            total = as.integer(total), 
            mouse_id = factor(mouse_id, levels=rownames(K2)), # levels must match, in order, for propto()
            dummy = factor(1),
            DOwave = as.factor(DOwave),
        )

    # For the binomial model: needs allele-specific counts
    au <- allele_unique |>
        filter(
            gene_id == gene,
        ) |>
        mutate(
            hap1 = str_sub(diplotype, 1, 1),
            hap2 = str_sub(diplotype, 2, 2),
            is_homozygous = hap1 == hap2,
        ) |>
        left_join(
            genotypes |>
                filter(gene_id == gene),
            "mouse_id",
        ) |>
        mutate(
            effect_A = (hap1 == "A") * A - (hap2 == "A") * A,
            effect_B = (hap1 == "B") * B - (hap2 == "B") * B,
            effect_C = (hap1 == "C") * C - (hap2 == "C") * C,
            effect_D = (hap1 == "D") * D - (hap2 == "D") * D,
            effect_E = (hap1 == "E") * E - (hap2 == "E") * E,
            effect_F = (hap1 == "F") * F - (hap2 == "F") * F,
            effect_G = (hap1 == "G") * G - (hap2 == "G") * G,
            effect_H = (hap1 == "H") * H - (hap2 == "H") * H,
        ) |>
        left_join(
            phenotypes,
            join_by(mouse_id == mouse.id)
        ) |>
        left_join(size_factors, "mouse_id") |>
        mutate(
            total = as.integer(total_reads), 
            mouse_id = factor(mouse_id, levels=rownames(K2)), # levels must match, in order, for propto()
            dummy = factor(1),
            DOwave = as.factor(DOwave),
        )


    res <- fit_model(au) |>
        mutate(
            gene_id = gene,
            .before=1,
        )
    temp[[length(temp)+1]] <- res
}
results <- bind_rows(temp)

results |> write_tsv(outfile)
