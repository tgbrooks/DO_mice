import numpy as np
import polars as pl
import scipy.stats

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
    type=pl.col("type").cast(pl.Enum(["no_cis", "no_buffering", "buffering"]))
)

haplotypes = list("ABCDEFGH")

# Check if the binomial step is performing well
# True and estimated haplotype effects should be well-correlated
temp = []
for hap in haplotypes:
    if hap == "H":
        continue
    # H is used as the reference in the model, so we compare everything to that
    est = data[f"effect_{hap}"]
    true = data[f"effect_{hap}_true"] - data["effect_H_true"]
    res = scipy.stats.linregress(
        true,
        est,
    )
    corr = np.corrcoef(est, true)[0, 1]
    temp.append(
        {
            "haplotype": hap,
            "intercept": res.intercept,
            "slope": res.slope,
            "correlation": corr,
        }
    )
binom_test = pl.DataFrame(temp)

print("""
---------------------------------------------------------------------
BINOMIAL MODEL:
---------------------------------------------------------------------
Here we check true and estimated parameters of the binomial model
We want correlation close to 1, intercept close to 0, and slope
close to 1.
""")
print(binom_test)
print(
    "We want binomial p-values to be small only for the genes that have a cis haplotype effect"
)
print(
    data.group_by("type")
    .agg(
        median_p=pl.col("anova_binom_p").median(),
        median_chisq=pl.col("anova_binom_chisq").median(),
    )
    .sort("type")
)


#### CHECK BUFFERING MODEL
print("""
---------------------------------------------------------------------
BUFFERING MODEL:
---------------------------------------------------------------------
Here we check if the buffering model performs as expected. Three types
of genes were simulated: ones with no cis haplotype effect at all, with
cis haplotype effects but no buffering, and those with both cis haplotype
effects and buffering.
""")
print(
    data.group_by("type")
    .agg(
        median_p=pl.col("anova_buffering_p").median(),
        median_chisq=pl.col("anova_buffering_chisq").median(),
    )
    .sort("type")
)
print("Estimated buffering factors:")
print(
    data.group_by("type")
    .agg(
        min_factor=pl.col("buffering_factor").min(),
        median_factor=pl.col("buffering_factor").median(),
        max_factor=pl.col("buffering_factor").max(),
    )
    .sort("type")
)
