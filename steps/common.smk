"""Shared configuration, sample tables, and helpers for the DO mouse pipeline.

Sample tables live in `samples/{tissue}.tsv` and are produced before the run by
`scripts/setup_geo_samples.py` (which resolves a GEO series into SRA runs). Each
row is one sequencing run:

    mouse_id  geo_sample  experiment  run  layout  [sex]  [generation]

Mice are selected per tissue in sorted order, restricted to the mice that have
genotypes when `geno/genotyped_mice.txt` exists, and capped at
`config["max_samples"]`.
"""

import csv
import os
import sys
import pathlib

TISSUES = list(config["run"])
HAPLOTYPES = config["haplotypes"]
HAP_LIST = HAPLOTYPES.split(",")

SAMPLES_DIR = "samples"
GENOTYPED_MICE = "geno/genotyped_mice.txt"

GBRS = config["gbrs"]
GBRS_DIR = GBRS["data_dir"]


def gbrs_file(key: str) -> str:
    """Path to one file of the GBRS supporting-files bundle."""
    return os.path.join(GBRS_DIR, GBRS[key])


BOWTIE_INDEX = os.path.join(GBRS_DIR, GBRS["bowtie_index"])
BOWTIE_INDEX_FILES = [
    f"{BOWTIE_INDEX}.{part}.{GBRS['bowtie_index_ext']}"
    for part in ("1", "2", "3", "4", "rev.1", "rev.2")
]


def _load_samples(tissue: str) -> list[dict]:
    """Read one tissue's sample table, or return [] if it hasn't been made yet."""
    path = f"{SAMPLES_DIR}/{tissue}.tsv"
    if not os.path.exists(path):
        print(
            f"WARNING: {path} not found; no samples for tissue {tissue}. Run\n"
            f"    python3 scripts/setup_geo_samples.py --tissue {tissue} "
            f"--geo {config['tissues'][tissue]['geo']}\n"
            "before running the pipeline (it needs internet access).",
            file=sys.stderr,
        )
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    missing = {"mouse_id", "run", "layout"} - set(rows[0] if rows else {})
    if rows and missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    return rows


SAMPLE_ROWS: dict[str, list[dict]] = {t: _load_samples(t) for t in TISSUES}

# run accession -> True if paired-end. Run accessions are unique across tissues.
RUN_PAIRED: dict[str, bool] = {}
# (tissue, mouse) -> list of run accessions, in table order.
MOUSE_RUNS: dict[tuple[str, str], list[str]] = {}
# mouse -> row, for per-mouse metadata (sex, DO generation) shared across tissues.
MOUSE_META: dict[str, dict] = {}

for _tissue, _rows in SAMPLE_ROWS.items():
    for _row in _rows:
        _run = _row["run"].strip()
        _mouse = _row["mouse_id"].strip()
        if not _run or not _mouse:
            continue
        RUN_PAIRED[_run] = _row["layout"].strip().upper() == "PAIRED"
        MOUSE_RUNS.setdefault((_tissue, _mouse), []).append(_run)
        MOUSE_META.setdefault(_mouse, _row)


def _genotyped_mice() -> set[str] | None:
    """Mice present in the downloaded genotypes, or None if not yet known.

    `geno/genotyped_mice.txt` is written by the `list_genotyped_mice` rule. It
    is read here at parse time, so generate it first (see README) if you want
    sample selection to skip mice that have no genotype.
    """
    if not os.path.exists(GENOTYPED_MICE):
        return None
    with open(GENOTYPED_MICE) as f:
        return {line.strip() for line in f if line.strip()}


GENOTYPED = _genotyped_mice()
if GENOTYPED is None:
    print(
        f"NOTE: {GENOTYPED_MICE} not found, so mice are selected from the sample "
        "tables alone. Generate it first (snakemake genotyped_mice) to restrict "
        "the selection to mice that actually have genotypes.",
        file=sys.stderr,
    )


def selected_mice(tissue: str) -> list[str]:
    """The mice to process for one tissue, capped at config['max_samples']."""
    mice = sorted({m for (t, m) in MOUSE_RUNS if t == tissue})
    if GENOTYPED is not None:
        without_geno = [m for m in mice if m not in GENOTYPED]
        if without_geno:
            print(
                f"NOTE: {len(without_geno)} {tissue} mice have no genotype and are "
                f"skipped, e.g. {', '.join(without_geno[:5])}",
                file=sys.stderr,
            )
        mice = [m for m in mice if m in GENOTYPED]
    limit = config.get("max_samples")
    if limit is not None:
        mice = mice[:limit]
    return mice


MICE = {t: selected_mice(t) for t in TISSUES}
# Genotypes are per mouse and shared by every tissue that mouse appears in.
ALL_MICE = sorted({m for mice in MICE.values() for m in mice})


def is_paired(tissue: str, mouse: str) -> bool:
    """True if a mouse's runs are paired-end. Errors on a mix of layouts."""
    layouts = {RUN_PAIRED[run] for run in MOUSE_RUNS[(tissue, mouse)]}
    if len(layouts) > 1:
        raise ValueError(
            f"{tissue}/{mouse} mixes paired- and single-end runs "
            f"({', '.join(MOUSE_RUNS[(tissue, mouse)])}). Split it into separate "
            "mice or drop the odd runs from the sample table."
        )
    return layouts.pop()


def read_ends(tissue: str, mouse: str) -> list[str]:
    """FASTQ ends to align for a mouse: R1 + R2 if paired, else SE."""
    return ["R1", "R2"] if is_paired(tissue, mouse) else ["SE"]


def transition_prob_file(mouse: str) -> str:
    """Transition-probability file for a mouse's DO generation and sex.

    Only used by `gbrs reconstruct`. Per-mouse `generation` / `sex` columns in
    the sample table win over the config defaults.
    """
    meta = MOUSE_META.get(mouse, {})
    generation = (meta.get("generation") or "").strip() or GBRS["default_generation"]
    sex = (meta.get("sex") or "").strip() or GBRS["default_sex"]
    sex = "F" if sex.upper().startswith("F") else "M"
    name = GBRS["transition_prob"].format(generation=generation, sex=sex)
    return os.path.join(GBRS_DIR, name)


def genotype_file(tissue: str, mouse: str) -> str:
    """The genotype calls used for the diploid quantification of a mouse."""
    source = GBRS["genotype_source"]
    if source == "downloaded":
        return f"geno/gbrs_genotypes/{mouse}.genotypes.tsv"
    if source == "reconstructed":
        return f"results/{tissue}/gbrs/{mouse}.genotypes.tsv"
    raise ValueError(
        f"gbrs: genotype_source must be 'downloaded' or 'reconstructed', not {source!r}"
    )


wildcard_constraints:
    # SRA/ENA/DDBJ run accessions, e.g. SRR1234567. The trailing digits keep this
    # from also matching the paired FASTQ suffixes (_1 / _2).
    run = r"[SED]RR[0-9]+",
    # Mouse IDs, e.g. DO021. No underscores, so `{mouse}_R1` is unambiguous.
    mouse = r"[A-Za-z0-9]+",
    tissue = '|'.join(TISSUES), # only real, biological tissues
    tissue_general = r"[A-Za-z0-9_]+",  # for when we want to include simulated data
    end = r"R1|R2|SE",
    # GBRS quantification modes (multi-way across founders, or diploid).
    mode = r"multiway|diploid",
