library(tidyverse)
library(ashr)

buffering_file <- "results/Adipose/buffering.txt"

buffering <- read_tsv(buffering_file)

simplify_nonzero <- function(ashr_fit, threshold=0.05) {
    # Fit this as if it's a single normal and a spike.
    # Send everything sufficiently small to exactly zero
    # We don't need it to be exact here
    nonnegligible <- ashr_fit$sd > threshold
    total_pi <-  sum(ashr_fit$pi[nonnegligible])
    mean_sd <- ashr_fit$pi[nonnegligible] * ashr_fit$sd[nonnegligible] / total_pi
    return list(sd=mean_sd, pi=total_pi)
}

# First, we change the reference to be the mean effect (on log scale)
baseline <- (
    + buffering$effect_A
    + buffering$effect_B
    + buffering$effect_C
    + buffering$effect_D
    + buffering$effect_E
    + buffering$effect_F
    + buffering$effect_G
) / 8
baseline_se <- sqrt(
    + buffering$effect_A_se^2
    + buffering$effect_B_se^2
    + buffering$effect_C_se^2
    + buffering$effect_D_se^2
    + buffering$effect_E_se^2
    + buffering$effect_F_se^2
    + buffering$effect_G_se^2
) / 8
equalized <- buffering |>
    mutate(
        effect_A = effect_A - baseline,
        effect_B = effect_B - baseline,
        effect_C = effect_C - baseline,
        effect_D = effect_D - baseline,
        effect_E = effect_E - baseline,
        effect_F = effect_F - baseline,
        effect_G = effect_G - baseline,
        effect_H = 0 - baseline,
        effect_A_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_B_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_C_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_D_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_E_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_F_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_G_se = sqrt((6/8)^2*effect_A_se^2 + baseline_se^2),
        effect_H_se = baseline_se,
    )

all_effects <- c(
    buffering$effect_A,
    buffering$effect_B,
    buffering$effect_C,
    buffering$effect_D,
    buffering$effect_E,
    buffering$effect_F,
    buffering$effect_G,
    buffering$effect_H,
)
all_effects_se <- c(
    buffering$effect_A_se,
    buffering$effect_B_se,
    buffering$effect_C_se,
    buffering$effect_D_se,
    buffering$effect_E_se,
    buffering$effect_F_se,
    buffering$effect_G_se,
    buffering$effect_G_se,
)

# Empirical bayes fit for allele-specific ratios
fit <- ash(all_effects, all_effects_se, mixcompdist="normal")
simple_fit <- simplify_nonzero(fit$fit)
simple_fit$sd <- simple_fit$sd
