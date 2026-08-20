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

results = pl.read_csv(
    "processed/simulated_counts/buffering/1.txt", separator="\t", null_values="NA"
)
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
for hap in haplotypes:
    if hap == "H":
        continue
    for model in data["model"].unique():
        # H is used as the reference in the model, so we compare everything to that
        d = data.filter(model=model)
        est = d[f"effect_{hap}"]
        true = d[f"effect_{hap}_true"] - d["effect_H_true"]
        res = scipy.stats.linregress(
            true,
            est,
        )
        corr = np.corrcoef(est, true)[0, 1]
        temp.append(
            {
                "haplotype": hap,
                "model": model,
                "intercept": res.intercept,
                "slope": res.slope,
                "correlation": corr,
            }
        )
binom_test = pl.DataFrame(temp).sort("model", "haplotype")


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
        fraction_failed_binom=pl.col("converged_binom").mean(),
        fraction_failed_buffering=pl.col("converged_buffering").mean(),
    )
    .sort("type", "model")
)
data = data.filter(pl.col("converged_binom") == 0, pl.col("converged_buffering") == 0)

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


#### CHECK BUFFERING MODEL
print("""
---------------------------------------------------------------------
BUFFERING MODEL:
---------------------------------------------------------------------
Here we check if the buffering model performs as expected.
""")
print(
    data.group_by("type", "model")
    .agg(
        median_p=pl.col("anova_buffering_p").median(),
        median_chisq=pl.col("anova_buffering_chisq").median(),
    )
    .sort("type", "model")
)
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

print(
    "Check if the estimated buffering factor corresponds with the actual buffering effect"
)
print("  (Only for the buffering genes) ")
buff = data.filter(type="buffering")
res = scipy.stats.linregress(buff["buffering_effect"], buff["buffering_factor"])
print(
    pl.DataFrame(
        {
            "slope": res.slope,
            "intercept": res.intercept,
            "pvalue": res.pvalue,
        }
    )
)

## AUC for identifying buffering
roc = (
    data.drop_nulls("anova_buffering_p")
    .sort("anova_buffering_p")
    .select(
        p_value="anova_buffering_p",
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
print(f"AUC of ROC curve: {auc_roc:0.3}")
