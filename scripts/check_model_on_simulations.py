import pathlib
import numpy as np
import polars as pl
import scipy.stats
import scipy.integrate

pl.Config(
    tbl_hide_dataframe_shape=True,
    tbl_rows=25,
    tbl_cols=20,
    tbl_hide_column_data_types=True,
)

temp = []
for file in pathlib.Path("processed/simulated_counts/buffering").glob("*.txt"):
    temp.append(
        pl.read_csv(
            file,
            separator="\t",
            null_values="NA",
        )
    )
results = pl.concat(temp)
truth = pl.read_csv("processed/simulated_counts/true_params.txt", separator="\t")

data = results.join(truth, "gene_id", suffix="_true").with_columns(
    type=pl.col("type").cast(pl.Enum(["no_cis", "no_buffering", "buffering"])),
    model=pl.col("model").cast(
        pl.Enum(["POISSON", "NEGATIVE_BINOMIAL", "SHARED_DISPERSION"])
    ),
)

haplotypes = list("ABCDEFGH")

# Check if the binomial step is performing well
# True and estimated haplotype effects should be well-correlated
temp = []
for model in data["model"].unique():
    est = []
    true = []
    for hap in haplotypes:
        if hap == "H":
            continue
        # H is used as the reference in the model, so we compare everything to that
        d = data.filter(model=model)
        est.append(d[f"effect_{hap}"].to_numpy())
        true.append((d[f"effect_{hap}_true"] - d["effect_H_true"]).to_numpy())
    est = np.concatenate(est)
    true = np.concatenate(true)
    res = scipy.stats.linregress(
        true,
        est,
    )
    corr = np.corrcoef(est, true)[0, 1]
    temp.append(
        {
            "model": model,
            "intercept": res.intercept,
            "slope": res.slope,
            "correlation": corr,
        }
    )
binom_test = pl.DataFrame(temp).sort("model")


print("""
---------------------------------------------------------------------
CONVERGENCE
---------------------------------------------------------------------
First, we check how many models converged and discard any that didn't
""")
# NOTE: 0 denotes successful convergence from glmmTMB
print(
    data.group_by("type", "model")
    .agg(
        fraction_failed_binom=(pl.col("convergence_code_binom") != 0).mean(),
        fraction_failed_total=(pl.col("convergence_code_total") != 0).mean(),
    )
    .sort("type", "model")
)
data = data.filter(
    pl.col("convergence_code_binom") == 0, pl.col("convergence_code_total") == 0
)

print("""
---------------------------------------------------------------------
BINOMIAL MODEL:
---------------------------------------------------------------------
Here we check true and estimated parameters of the binomial model
We want correlation close to 1, intercept close to 0, and slope
close to 1.
Three types of genes were simulated: ones with no cis haplotype effect at all, with
cis haplotype effects but no buffering, and those with both cis haplotype
effects and buffering.
We also simulated under three allele-expression models:
1. Poisson: both alleles independent Poissons,
2. Negative binomial: both alleles independent negative binomials,
3. Shared dispersion: marginally negative binomials but dispersion factor
    is first drawn for both of them, so not independent.
""")
print(binom_test)
print(
    "We want binomial p-values to be small only for the genes that have a cis haplotype effect"
)
print(
    data.group_by("type", "model")
    .agg(
        median_p=pl.col("anova_binom_p").median(),
        median_chisq=pl.col("anova_binom_chisq").median(),
    )
    .sort("type", "model")
)

print("Check that fit effects match simulated effects")


#### CHECK TOTAL COUNTS MODEL
print("""
---------------------------------------------------------------------
NEGATIVE BINOMIAL (TOTAL COUNTS) MODEL:
---------------------------------------------------------------------
Here we check that true and estimated parameters of the negative binomial
model of total counts. We want correlation close to 1, intercept 0,
and slope 1. Comparing true and estimated values.

Among non-buffering genes:
""")

# Check if the NB step is performing well
# True and estimated haplotype effects should be well-correlated
temp = []
for model in data["model"].unique():
    est = []
    true = []
    for hap in haplotypes:
        if hap == "H":
            continue
        # H is used as the reference in the model, so we compare everything to that
        d = data.filter(pl.col("type") != "buffering", model=model)
        est.append(d[f"total_{hap}"].to_numpy())
        true.append((d[f"effect_{hap}_true"] - d["effect_H_true"]).to_numpy())
    true = np.concatenate(true)
    est = np.concatenate(est)
    res = scipy.stats.linregress(
        true,
        est,
    )
    corr = np.corrcoef(true, est)[0, 1]
    temp.append(
        {
            "model": model,
            "intercept": res.intercept,
            "slope": res.slope,
            "correlation": corr,
        }
    )
nb_test = pl.DataFrame(temp).sort("model")
print(nb_test)

#### CHECK BUFFERING MODEL
print("""
---------------------------------------------------------------------
BUFFERING MODEL:
---------------------------------------------------------------------
Here we check if the buffering model performs as expected.
""")
print("Estimated buffering factors:")
print(
    data.group_by("type", "model")
    .agg(
        min_factor=pl.col("buffering_factor").min(),
        median_factor=pl.col("buffering_factor").median(),
        max_factor=pl.col("buffering_factor").max(),
    )
    .sort("type", "model")
)
print("Confidence intervals:")
print(
    data.group_by("type", "model")
    .agg(
        hi_min_factor=pl.col("buffering_factor_ci_hi").min(),
        hi_median_factor=pl.col("buffering_factor_ci_hi").median(),
        hi_max_factor=pl.col("buffering_factor_ci_hi").max(),
        lo_min_factor=pl.col("buffering_factor_ci_lo").min(),
        lo_median_factor=pl.col("buffering_factor_ci_lo").median(),
        lo_max_factor=pl.col("buffering_factor_ci_lo").max(),
        num_significant=(pl.col("buffering_factor_ci_hi") < 1).mean(),
    )
    .sort("type", "model")
)

print(
    "Check if the estimated buffering factor corresponds with the actual buffering effect"
)
print("  (Only for the buffering genes) ")
temp = []
for (model,), _dat in data.group_by("model"):
    buff = _dat.filter(type="buffering")
    res = scipy.stats.linregress(
        buff["true_buffering_factor"], buff["buffering_factor"]
    )
    corr = np.corrcoef(buff["true_buffering_factor"], buff["buffering_factor"])[0, 1]

    ## AUC for identifying buffering
    roc = (
        _dat.sort("buffering_factor_ci_hi")
        .select(
            statistic="buffering_factor_ci_hi",
            n_below=pl.row_index() + 1,
            n_true_positives=(pl.col("type") == "buffering").cum_sum(),
            n_positives=(pl.col("type") == "buffering").sum(),
        )
        .with_columns(
            fpr=(pl.col("n_below") - pl.col("n_true_positives"))
            / (pl.len() - pl.col("n_positives")),
            tpr=pl.col("n_true_positives") / pl.col("n_positives"),
        )
    )
    auc_roc = scipy.integrate.trapezoid(roc["tpr"], roc["fpr"])
    temp.append(
        {
            "model": model,
            "slope": res.slope,
            "intercept": res.intercept,
            "pvalue": res.pvalue,
            "correlation": corr,
            "ROC AUC": auc_roc,
        }
    )
res = pl.DataFrame(temp)
print(res)
