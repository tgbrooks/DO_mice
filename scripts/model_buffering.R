library(tidyverse)
library(glmmTMB)
library(arrow)


count_file <- "results/Adipose/Adipose.diploid.genes.founder_expected_read_counts.parquet"
chromosome <- "5"
MIN_MEDIAN_COUNTS <- 50
size_factors_file <- "results/Adipose/size_factors.txt"
annot_file <- "processed/gene_annot.txt"
outfile <- "temp.txt"
kinship_file <- paste0("geno/kinship/", chromosome, ".txt")
phenotypes_file <- "phenotypes.csv.gz"
allele_unique_reads <- "processed/Adipose/allele_unique_reads.parquet"

count_file <- "processed/simulated_counts/simulated_counts.diploid.genes.founder_expected_read_counts.parquet"
chromosome <- "1"
MIN_MEDIAN_COUNTS <- 50
size_factors_file <- "processed/simulated_counts/size_factors.txt"
annot_file <- "processed/simulated_counts/gene_annot.txt"
kinship_file <- "processed/simulated_counts/kinship.txt"
phenotypes_file <- "processed/simulated_counts/phenotypes.csv"
allele_unique_reads <- "processed/simulated_counts/allele_unique_reads.parquet"
outfile <- "temp.txt"


count_file <- snakemake@input$counts
size_factors_file <- snakemake@input$size_factors
annot_file <- snakemake@input$annot
kinship_file <- snakemake@input$kinship
phenotypes_file <- snakemake@input$phenotypes
allele_unique_reads <- snakemake@input$allele_unique_reads
chromosome <- snakemake@wildcards$chromosome
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
allele_unique <- read_parquet(allele_unique_reads) |>
        mutate(
            hap1 = str_sub(diplotype, 1, 1),
            hap2 = str_sub(diplotype, 2, 2),
            is_homozygous = hap1 == hap2,
            A = str_count(diplotype, "A") ,
            B = str_count(diplotype, "B") ,
            C = str_count(diplotype, "C") ,
            D = str_count(diplotype, "D") ,
            E = str_count(diplotype, "E") ,
            F = str_count(diplotype, "F") ,
            G = str_count(diplotype, "G") ,
            H = str_count(diplotype, "H") ,
        )

size_factors <- read_tsv(size_factors_file)
phenotypes <- read_csv(phenotypes_file)

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
    family = betabinomial
    res_binom <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ effect_A + effect_B + effect_C + effect_D + effect_E + effect_F + effect_G,  # H is reference
        family = family,
        data = au2,
    )
    binom_disp <- exp(res_binom$fit$par['betadisp'])
    if ((res_binom$fit$convergence > 0) && (is.na(binom_disp) || (binom_disp > 100))) {
        # High 'dispersion' parameter in the betabinomial parameterization used by glmmTMB
        # means that it converges to a standard binomial. Use that instead.
        family <- binomial
        res_binom <- glmmTMB(
            cbind(haplotype_1_unique, haplotype_2_unique) ~ effect_A + effect_B + effect_C + effect_D + effect_E + effect_F + effect_G,  # H is reference
            family = family,
            data = au2,
        )
    }
    res_binom_restricted <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ 1,
        family = family,
        data = au2,
    )
    anova_binom <- anova(res_binom_restricted, res_binom)

    cis_A <- fixef(res_binom)$cond['effect_A']
    cis_B <- fixef(res_binom)$cond['effect_B']
    cis_C <- fixef(res_binom)$cond['effect_C']
    cis_D <- fixef(res_binom)$cond['effect_D']
    cis_E <- fixef(res_binom)$cond['effect_E']
    cis_F <- fixef(res_binom)$cond['effect_F']
    cis_G <- fixef(res_binom)$cond['effect_G']

    v <- vcov(res_binom)$cond
    cis_A_se <- sqrt(v['effect_A', 'effect_A'])
    cis_B_se <- sqrt(v['effect_B', 'effect_B'])
    cis_C_se <- sqrt(v['effect_C', 'effect_C'])
    cis_D_se <- sqrt(v['effect_D', 'effect_D'])
    cis_E_se <- sqrt(v['effect_E', 'effect_E'])
    cis_F_se <- sqrt(v['effect_F', 'effect_F'])
    cis_G_se <- sqrt(v['effect_G', 'effect_G'])

    ## Now we fit the buffering model, using the binomial-fit parameters
    ## We assume that tot = y_1  + y_2 is buffered by some function of itself
    ## log(tot) = cis_effect + b log(tot)
    ## log(tot) = k * cis_effect
    ## where k = 1/(1-b)
    ## and b gives the extent of buffering
    ## and we know the total cis effect from the binomial model above.
    ## To compute the cis effect, we model log(total)] = log(y1 + y2) the sum of the two alleles counts (not just unique)
    ## From our binomial model, we have E[y1] = exp(effect_i) and where i is the first haplotype
    ## So log(total) ≈  log(exp(effect_i) + exp(effect_j))
    buffering_data <- au |>
        mutate(
            cis_effect = log(
                   exp(cis_A)*A
                 + exp(cis_B)*B
                 + exp(cis_C)*C
                 + exp(cis_D)*D
                 + exp(cis_E)*E
                 + exp(cis_F)*F
                 + exp(cis_G)*G
                 + H
                 # H is reference, effect_H = 0
            )
        )

    if (all(is.na(buffering_data$cis_effect))) {
        message("Skipping ", gene, " : estimated cis effects were NA")
        message("Diplotype distribution in allele-uniques:")
        print(au2$diplotype |> table())
        return("SKIP")
    }

    ## NOTE: one limitation of this estimate is that there is noise in the exogenous variable
    ## cis_effect, which can dilute the slope and therefore make it look like buffering.
    # NOTE: we don't include generation since it's not in the available phenotypes
    res_buffering <- glmmTMB(
        total ~ offset(log(size_factor)) + sex + DOwave +
                cis_effect +
                propto(0 + mouse_id | dummy, K2),
        family = nbinom2,
        data = buffering_data,
    )
    # Including `cis_effect` in offset() fixes its coefficient at 1, our null
    res_buffering_restricted <- glmmTMB(
        total ~ offset(log(size_factor) + cis_effect) + sex + DOwave +
                propto(0 + mouse_id | dummy, K2),
        family = nbinom2,
        data = buffering_data,
    )
    anova_buffering = anova(res_buffering_restricted, res_buffering)

    buffering_factor <- fixef(res_buffering)$cond['cis_effect']
    buffering_factor_se <- sqrt(vcov(res_buffering)$cond['cis_effect', 'cis_effect'])
    results <- tibble(
        buffering_factor = buffering_factor,
        bufferring_factor_se = buffering_factor_se,
        dilution_factor = dilution_factor,
        downsample_factor = downsample_factor,
        n_samples_binom = nrow(au2),
        n_samples_buffering = nrow(buffering_data),
        convergence_code_binom = res_binom$fit$convergence,
        convergence_code_buffering = res_buffering$fit$convergence,
        effect_A = cis_A,
        effect_B = cis_B,
        effect_C = cis_C,
        effect_D = cis_D,
        effect_E = cis_E,
        effect_F = cis_F,
        effect_G = cis_G,
        effect_H = 0, # Reference, 0 by definition
        effect_A_se = cis_A_se,
        effect_B_se = cis_B_se,
        effect_C_se = cis_C_se,
        effect_D_se = cis_D_se,
        effect_E_se = cis_E_se,
        effect_F_se = cis_F_se,
        effect_G_se = cis_G_se,
        effect_H_se = 0, # Reference, 0 by definition
        anova_binom_p = anova_binom$`Pr(>Chisq)`[2],
        anova_binom_chisq = anova_binom$Chisq[2],
        anova_buffering_p = anova_buffering$`Pr(>Chisq)`[2],
        anova_buffering_chisq = anova_buffering$Chisq[2],
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

    # For the binomial model: needs allele-specific counts
    au <- allele_unique |>
        filter(
            gene_id == gene,
        ) |>
        mutate(
            effect_A = (hap1 == "A") - (hap2 == "A"),
            effect_B = (hap1 == "B") - (hap2 == "B"),
            effect_C = (hap1 == "C") - (hap2 == "C"),
            effect_D = (hap1 == "D") - (hap2 == "D"),
            effect_E = (hap1 == "E") - (hap2 == "E"),
            effect_F = (hap1 == "F") - (hap2 == "F"),
            effect_G = (hap1 == "G") - (hap2 == "G"),
            effect_H = (hap1 == "H") - (hap2 == "H"),
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

    res <- fit_model(au)
    if (identical(res, "SKIP")) {
        next
    }

    res <- res |>
        mutate(
            gene_id = gene,
            .before=1,
        )
    temp[[length(temp)+1]] <- res
}
results <- bind_rows(temp)

results |> write_tsv(outfile)
