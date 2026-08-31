library(tidyverse)
library(glmmTMB)
library(arrow)


count_file <- "results/Adipose/Adipose.diploid.genes.founder_expected_read_counts.parquet"
MIN_MEDIAN_COUNTS <- 50
size_factors_file <- "results/Adipose/size_factors.txt"
outfile <- "temp.txt"
chromosome <- "5"
kinship_file <- paste0("geno/kinship/", chromosome, ".txt")
phenotypes_file <- "phenotypes.csv.gz"
gene_ids <- c(
    "ENSMUSG00000018845",
    "ENSMUSG00000031902", # highly buffered
    "ENSMUSG00000025337" # minimal cis effects
)
outlier_ids <- c("DO122")
allele_unique_reads <- "processed/Adipose/allele_unique_reads.parquet"

count_file <- "processed/simulated_counts/simulated_counts.diploid.genes.founder_expected_read_counts.parquet"
MIN_MEDIAN_COUNTS <- 50
size_factors_file <- "processed/simulated_counts/size_factors.txt"
kinship_file <- "processed/simulated_counts/kinship.txt"
phenotypes_file <- "processed/simulated_counts/phenotypes.csv"
allele_unique_reads <- "processed/simulated_counts/allele_unique_reads.parquet"
outlier_ids <- c()
gene_ids <- c("GENE0000", "GENE0001", "GENE00002")
outfile <- "temp.txt"


count_file <- snakemake@input$counts
size_factors_file <- snakemake@input$size_factors
kinship_file <- snakemake@input$kinship
phenotypes_file <- snakemake@input$phenotypes
allele_unique_reads <- snakemake@input$allele_unique_reads
MIN_MEDIAN_COUNTS <- snakemake@params$min_median_counts
gene_ids <- snakemake@params$genes
outlier_ids <- snakemake@params$outlier_ids
outfile <- snakemake@output$outfile


counts <- read_parquet(count_file)
mouse_ids <- unique((counts |> filter(!(mouse_id %in% outlier_ids)))$mouse_id)

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
        ) |>
        filter( # Discard outliers
            !(mouse_id %in% outlier_ids),
        )

size_factors <- read_tsv(size_factors_file)
phenotypes <- read_csv(phenotypes_file)

covariance_deming_regression <- function(b1, b2, V1, V2, C = NULL, grid = seq(-2, 3, length = 2001)) {
    # Function written by Claude Opus 5 to perform Deming-like regression when we have
    # two k-vectors beta_1 and beta_2 each with known covariance matrices V_1 and V_2
    # We assume no cross-covariance between beta_1 and beta_2 (when using C = 0, default)
    # Model: beta_1 = theta + epsilon_1
    #        beta_2 = lambda * theta + epsilon_2
    #        epsilon_i ~ N(0, V_i)
    # So theta is the underlying 'true' parameter vector and lambda is the scaling between the two
    #
    # Q(lambda) := d(lambda)^T S(lambda)^-1 d(lambda)
    #   d(lambda) := beta_2 - lambda beta_1
    #   S(lambda) := V_2 + lambda^2 V_1
    # so Q(lambda) is profile deviance (theta is profiled out) given covariances.
    # Our solution is then the value of lambda that minimizes Q.
    # If cross-covariance (C != 0), then we use instead:
    #   S(lambda) := V_2 + lambda^2 V_1 - lambda(C + C')
    p <- length(b1)
    if (is.null(C)) C <- matrix(0, p, p) # C is cross-covariance between the betas
    Cs <- C + t(C)
    Q <- function(lam) {
        # objective function
        d <- b2 - lam * b1
        S <- V2 + lam^2 * V1 - lam * Cs
        e <- eigen(S, symmetric = TRUE)
        k <- e$values > max(e$values) * 1e-8          # generalized inverse
        sum((crossprod(e$vectors[, k, drop = FALSE], d)^2) / e$values[k])
    }
    # First minimize Q over a discrete grid
    q   <- vapply(grid, Q, 0)
    i   <- which.min(q)
    # Then find the minimum by numeric search in the neighborhood of that grid point
    opt <- optimize(Q, grid[c(max(i - 1, 1), min(i + 1, length(grid)))])
    # Now compute a profile confidence interval on lambda
    keep <- q <= opt$objective + qchisq(0.95, 1)
    r <- qr(V1 + V2)$rank
    list(
         lambda = opt$minimum,
         Q = opt$objective,
         df = r - 1,
         p_gof = pchisq(opt$objective, r - 1, lower.tail = FALSE),
         lo = min(grid[keep]),
         hi = max(grid[keep]),
         bounded = !(keep[1] || keep[length(keep)])
    )
}

compare_to_null_binom <- function(au2, family, ll_full) {
    # glmmTMB crashes if you give it an empty binomial model with no predictors
    # (but not betabinomial). But that's our null hypothesis. So we calculate it
    # exaclty here (nothing to fit with no parameters).
    y  <- au2$haplotype_1_unique
    m  <- y + au2$haplotype_2_unique
    ll0 <- if (identical(family, binomial)) {
        sum(dbinom(y, m, 0.5, log = TRUE))
    } else {                      # betabinomial: mu fixed at 0.5, phi free
        -optimize(function(lp) { a <- exp(lp)/2
            -sum(lchoose(m, y) + lbeta(y + a, m - y + a) - lbeta(a, a)) },
        c(-10, 30))$objective
    }
    chisq_binom <- 2 * (as.numeric(ll_full) - ll0)
    p_binom     <- pchisq(chisq_binom, df = 7, lower.tail = FALSE)
    return(list(pvalue=p_binom, chisq=chisq_binom))
}

fit_model <- function(au) {
    # Fit our full buffering model
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
        filter((haplotype_1_unique > 0) | (haplotype_2_unique > 0)) |>
        mutate(
            # Ensure sorted diplotype (always true for real data but not for simulation)
            diplo = paste0(pmin(hap1, hap2), pmax(hap1, hap2)),
            sgn  = ifelse(hap1 < hap2, 1L, -1L)
        )
    # Model for the goodness of fit test
    signed_diplo <- model.matrix(~ 0 + diplo, au2) * au2$sgn

    if (nrow(au2) == 0) {
        message("Skipping: no allele-specific expression")
        return("SKIP")
    }
    # Fit a logit binomial model for the two haplotype unique counts
    # this estimates the cis effects in a manner that is independent of buffering
    # since buffering affects both haplotypes
    family = betabinomial
    res_binom <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ 0 + effect_A + effect_B + effect_C + effect_D + effect_E + effect_F + effect_G,  # H is reference
        family = family,
        data = au2,
    )
    binom_disp <- exp(res_binom$fit$par['betadisp'])
    if ((res_binom$fit$convergence > 0) && (is.na(binom_disp) || (binom_disp > 100))) {
        # High 'dispersion' parameter in the betabinomial parameterization used by glmmTMB
        # means that it converges to a standard binomial. Use that instead.
        family <- binomial
        res_binom <- glmmTMB(
            cbind(haplotype_1_unique, haplotype_2_unique) ~ 0 + effect_A + effect_B + effect_C + effect_D + effect_E + effect_F + effect_G,  # H is reference
            family = family,
            data = au2,
        )
    }
    anova_binom <- compare_to_null_binom(au2, family, logLik(res_binom))

    # Goodness of fit test for the binomial model compared to modelling each diplotype separately
    res_binom_diplo <- glmmTMB(
        cbind(haplotype_1_unique, haplotype_2_unique) ~ 0 + signed_diplo,
        family = family,
        data = au2,
    )
    binom_gof <- anova(res_binom_diplo, res_binom)

    if (any(is.na(fixef(res_binom)$cond)) || any(is.na(vcov(res_binom)$cond))) {
        message("Skipping due to NA parameters in binom")
        return("SKIP")
    }

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

    # Fit the model on total counts (not allele-specific)
    res_total <- glmmTMB(
        total ~ offset(log(size_factor)) + sex + DOwave +
                A + B + C + D + E + F + G + # H is reference
                propto(0 + mouse_id | dummy, K2),
        family = nbinom2,
        data = au,
    )

    binom_vars <- c("effect_A", "effect_B", "effect_C", "effect_D", "effect_E", "effect_F", "effect_G")
    binom_effects <- fixef(res_binom)$cond[binom_vars]
    binom_cov <- vcov(res_binom)$cond[binom_vars, binom_vars]
    # NOTE: to compare binomial and total models, we scale by 2 since total model
    # is for the sum of two alleles: homozygous AA should have the same log fold change
    # in total over BB as A has over B in AB binomial model, if no buffering. But AA is
    # coded as A = 2, B = 0.
    total_vars <- c("A", "B", "C", "D", "E", "F", "G")
    total_effects <- fixef(res_total)$cond[total_vars] * 2
    total_cov <- vcov(res_total)$cond[total_vars, total_vars] * 4

    if (any(is.na(fixef(res_total)$cond)) || any(is.na(vcov(res_total)$cond))) {
        message("Skipping due to NA parameters in total")
        return("SKIP")
    }


    # Compare the outputs of these models via Deming-like regression
    deming <- covariance_deming_regression(binom_effects, total_effects, binom_cov, total_cov)


    results <- tibble(
        # From the deming regression
        buffering_factor = deming$lambda,
        buffering_factor_ci_lo = deming$lo,
        buffering_factor_ci_hi = deming$hi,
        buffering_factor_bounded = deming$bounded,
        deming_Q = deming$Q,
        deming_p_gof = deming$p_gof, # goodness of fit test
        # From the binomial model
        anova_binom_p = anova_binom$pvalue,
        anova_binom_chisq = anova_binom$chisq,
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
        binom_p_gof = binom_gof['res_binom_diplo', 'Pr(>Chisq)'],
        # From the total counts model
        total_A = total_effects['A'],
        total_B = total_effects['B'],
        total_C = total_effects['C'],
        total_D = total_effects['D'],
        total_E = total_effects['E'],
        total_F = total_effects['F'],
        total_G = total_effects['G'],
        total_H = 0, # reference, 0 by definition
        total_A_se = total_cov["A","A"],
        total_B_se = total_cov["B","B"],
        total_C_se = total_cov["C","C"],
        total_D_se = total_cov["D","D"],
        total_E_se = total_cov["E","E"],
        total_F_se = total_cov["F","F"],
        total_G_se = total_cov["G","G"],
        total_H_se = 0, # reference, 0 by definition
        dispersion = 1/exp(res_total$fit$par['betadisp']),
        # Meta data
        n_samples_binom = nrow(au2),
        n_samples_total = nrow(au),
        convergence_code_binom = res_binom$fit$convergence,
        convergence_code_total = res_total$fit$convergence,
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
