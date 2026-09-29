"""Convert one mouse's founder haplotype probabilities into GBRS genotype calls.

`gbrs quantify -G` takes a two-column file of per-gene diplotype calls:

    #Gene_ID	Diplotype
    ENSMUSG00000000001	DF
    ENSMUSG00000000003	CC

This script produces that file from array-based founder probabilities (exported
from the genoprobs .RData file by scripts/export_genoprobs.R), so that RNA-seq
can be quantified against each mouse's known genome instead of one reconstructed
from the expression data. A second file (--out-confidence) gives each call's
confidence and the number of markers behind it.

Genes are placed in bp using --gtf, which should be the Ensembl build of the
GBRS reference. Each gene is called from a window of markers: the nearest marker
upstream of the gene start, every marker within the gene, and the nearest marker
downstream of the gene end.

Each marker's founder probabilities are converted to founder dosages (summing
to 2) and called as homozygous when the top founder's dosage reaches
--hom-dosage-threshold, and as the heterozygous combination of the top two
founders otherwise. The gene's call is the most common marker call in the
window; ties go to the call that fits the window best (see below), then to the
call of the marker nearest the gene midpoint.

The confidence of a gene's call is the mean, over the window's markers, of how
well that marker supports the call: the fraction of the marker's founder dosage
that the called diplotype accounts for, sum_h min(dosage_h, call_dosage_h) / 2.
A marker that is certainly FF supports FF with 1, AF with 0.5, and AB with 0; a
marker at 75% F / 25% A supports FF with 0.75 and AF with 0.75. So the
confidence is 1 only when every marker in the window is certain of the called
diplotype, and falls with both disagreeing markers (e.g. a recombination within
the window) and markers with intermediate probabilities.

The GBRS gene position file is not used: its cM positions do not line up with
the genome grid's genetic map closely enough to find a gene's nearest markers.

Marker bp positions come, in order of preference, from the marker name when it
encodes a position (e.g. `1_3000000`), from the GBRS genome grid (matching
marker names), or from the marker table written by export_genoprobs.R.

Only numpy is required, so this runs inside the GBRS container.
"""

import argparse
import gzip
import re
import sys
from collections import Counter, OrderedDict

import numpy as np

MARKER_POS_RE = re.compile(r"^(?:chr)?([0-9]+|[XYMxym]|MT|mt)[_:.-]([0-9]+)$")
GTF_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')


def normalize_chrom(chrom: str) -> str:
    """Strip a `chr` prefix and normalize case, e.g. `chr1` -> `1`, `x` -> `X`."""
    chrom = chrom.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.upper() in ("X", "Y", "M", "MT"):
        return chrom.upper()
    return chrom


def read_alleleprobs(path: str, haplotypes: list[str]):
    """Read a per-mouse allele probability TSV written by export_genoprobs.R.

    Returns (markers, chroms, probs) where probs is (n_markers, n_haplotypes).
    """
    opener = gzip.open if path.endswith(".gz") else open
    markers: list[str] = []
    chroms: list[str] = []
    values: list[list[float]] = []
    with opener(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            marker_col = header.index("marker")
            chrom_col = header.index("chr")
        except ValueError:
            raise SystemExit(f"{path}: expected 'marker' and 'chr' columns, got {header}")
        try:
            hap_cols = [header.index(h) for h in haplotypes]
        except ValueError:
            raise SystemExit(
                f"{path}: missing founder column(s); expected {haplotypes}, got {header}"
            )
        for line in f:
            fields = line.rstrip("\n").split("\t")
            markers.append(fields[marker_col])
            chroms.append(normalize_chrom(fields[chrom_col]))
            values.append([float(fields[c]) for c in hap_cols])
    return np.array(markers), np.array(chroms), np.array(values, dtype=float)


def read_grid_bp(path: str | None) -> dict[str, float]:
    """Grid marker name -> bp position, from the GBRS genome grid.

    Columns are marker, chr, pos, cM[, bp]; when a separate bp column is
    absent, `pos` is the base-pair position.
    """
    if not path:
        return {}
    positions: dict[str, float] = {}
    with open(path) as f:
        first = f.readline()
        if not first.lower().startswith(("marker", "#")):
            f.seek(0)
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            positions[fields[0]] = float(fields[4]) if len(fields) > 4 else float(fields[2])
    return positions


def read_marker_table(path: str | None):
    """Read the optional marker position table from export_genoprobs.R."""
    if not path:
        return {}
    positions: dict[str, float] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        if "marker" not in header or "pos" not in header:
            return {}
        marker_col = header.index("marker")
        pos_col = header.index("pos")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(marker_col, pos_col):
                continue
            try:
                positions[fields[marker_col]] = float(fields[pos_col])
            except ValueError:
                continue
    return positions


def read_known_genes(path: str | None) -> set[str] | None:
    """Gene IDs in the EMASE gene-to-transcript file (its first column).

    `gbrs quantify` looks up every gene of the genotype file in this set, and
    fails on one it doesn't know, so calls are restricted to these genes.
    """
    if not path:
        return None
    genes = set()
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            genes.add(line.split("\t", 1)[0].strip())
    return genes


def read_gtf_genes(path: str):
    """Chromosome -> (gene IDs, starts, ends) from the `gene` lines of a GTF."""
    opener = gzip.open if path.endswith(".gz") else open
    rows: dict[str, list[tuple[str, float, float]]] = OrderedDict()
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = GTF_GENE_ID_RE.search(fields[8])
            if not match:
                continue
            rows.setdefault(normalize_chrom(fields[0]), []).append(
                (match.group(1), float(fields[3]), float(fields[4]))
            )
    genes = OrderedDict()
    for chrom, entries in rows.items():
        ids, starts, ends = zip(*entries)
        genes[chrom] = (np.array(ids), np.array(starts), np.array(ends))
    return genes


def marker_bp(
    markers: np.ndarray, grid_bp: dict, marker_table: dict, marker_units: str
) -> np.ndarray:
    """bp position of every marker; NaN when unknown."""
    bp = np.full(len(markers), np.nan)
    for i, name in enumerate(markers):
        match = MARKER_POS_RE.match(name)
        if match:
            bp[i] = float(match.group(2))
            continue
        if name in grid_bp:
            bp[i] = grid_bp[name]
            continue
        if name in marker_table:
            pos = marker_table[name]
            if marker_units == "bp":
                bp[i] = pos
            elif marker_units == "Mbp":
                bp[i] = pos * 1e6
            elif pos > 1e6:
                bp[i] = pos
            else:
                raise SystemExit(
                    "Cannot tell whether the marker positions in the marker table "
                    f"are bp or Mbp (value {pos}). Set genotypes: marker_units: to "
                    "'bp' or 'Mbp' in config.yaml."
                )
    return bp


def founder_dosages(probs: np.ndarray) -> np.ndarray | None:
    """Founder dosages (summing to 2) from one marker's probabilities."""
    total = probs.sum()
    if total <= 0:
        return None
    return 2.0 * probs / total


def call_diplotype(dosage: np.ndarray, hom_threshold: float) -> tuple[int, int]:
    """Call one marker's diplotype as a sorted pair of founder indices."""
    order = np.argsort(dosage)[::-1]
    top = int(order[0])
    if dosage[top] >= hom_threshold:
        return (top, top)
    return tuple(sorted((top, int(order[1]))))


def support(dosage: np.ndarray, pair: tuple[int, int]) -> float:
    """Fraction of a marker's founder dosage that a diplotype accounts for."""
    call_dosage = np.zeros_like(dosage)
    for i in pair:
        call_dosage[i] += 1.0
    return float(np.minimum(dosage, call_dosage).sum() / 2.0)


def call_gene(
    dosages: np.ndarray, marker_pos: np.ndarray, midpoint: float, hom_threshold: float
):
    """Call a gene from the dosages of the markers in its window.

    Returns (founder index pair, confidence). See the module docstring.
    """
    marker_calls = [call_diplotype(d, hom_threshold) for d in dosages]
    counts = Counter(marker_calls)
    mean_support = {
        pair: float(np.mean([support(d, pair) for d in dosages])) for pair in counts
    }
    central = marker_calls[int(np.argmin(np.abs(marker_pos - midpoint)))]
    best = max(counts, key=lambda p: (counts[p], mean_support[p], p == central))
    return best, mean_support[best]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alleleprobs", required=True, help="Per-mouse founder probability TSV")
    parser.add_argument(
        "--gtf", required=True, help="GTF of the GBRS reference build, for gene positions"
    )
    parser.add_argument("--grid", help="GBRS genome grid TSV, for marker positions")
    parser.add_argument("--markers", help="Marker position table from export_genoprobs.R")
    parser.add_argument(
        "--gene2transcripts",
        help="EMASE gene-to-transcript file; restricts calls to the genes GBRS knows",
    )
    parser.add_argument("--out", required=True, help="Output genotype calls TSV, for GBRS")
    parser.add_argument(
        "--out-confidence",
        help="Output TSV of each call's confidence and number of markers",
    )
    parser.add_argument("--sample", default="", help="Sample name, for log messages")
    parser.add_argument("--haplotypes", default="A,B,C,D,E,F,G,H")
    parser.add_argument(
        "--hom-dosage-threshold",
        type=float,
        default=1.5,
        help="Founder dosage (0-2) at or above which a marker is called homozygous",
    )
    parser.add_argument(
        "--min-call-prob",
        type=float,
        default=0.5,
        help="Report how many calls have a confidence below this",
    )
    parser.add_argument(
        "--marker-units",
        default="auto",
        choices=["auto", "bp", "Mbp"],
        help="Units of positions in the --markers table (only used as a fallback)",
    )
    args = parser.parse_args()

    haplotypes = args.haplotypes.split(",")
    markers, marker_chroms, probs = read_alleleprobs(args.alleleprobs, haplotypes)
    coords = marker_bp(
        markers, read_grid_bp(args.grid), read_marker_table(args.markers), args.marker_units
    )
    gtf_genes = read_gtf_genes(args.gtf)
    known_genes = read_known_genes(args.gene2transcripts)

    known = ~np.isnan(coords)
    if not known.any():
        raise SystemExit(
            "None of the marker positions could be resolved. The marker names in the "
            "genotype file match neither a `chr_position` pattern nor the GBRS genome "
            "grid, and no usable marker table was given. Supply positions via the "
            "map object in the .RData file (see scripts/export_genoprobs.R)."
        )
    if not known.all():
        print(
            f"WARNING: {int((~known).sum())} of {len(markers)} markers have no "
            "resolvable position and are ignored",
            file=sys.stderr,
        )

    calls: list[tuple[str, str, float, int]] = []
    n_missing_chrom = 0
    skipped_chroms: list[str] = []
    found_genes: set[str] = set()

    for chrom, (gene_ids, starts, ends) in gtf_genes.items():
        if known_genes is not None:
            keep = np.array([g in known_genes for g in gene_ids], dtype=bool)
            gene_ids, starts, ends = gene_ids[keep], starts[keep], ends[keep]
        if not len(gene_ids):
            continue
        found_genes.update(gene_ids)
        on_chrom = known & (marker_chroms == chrom)
        if not on_chrom.any():
            n_missing_chrom += len(gene_ids)
            skipped_chroms.append(chrom)
            continue
        order = np.argsort(coords[on_chrom])
        chrom_coords = coords[on_chrom][order]
        chrom_probs = probs[on_chrom][order]
        n_markers = len(chrom_coords)

        # Window: nearest marker before the start through nearest marker after
        # the end (a marker exactly at either end is within the gene).
        first = np.clip(np.searchsorted(chrom_coords, starts, side="left") - 1, 0, n_markers - 1)
        last = np.clip(np.searchsorted(chrom_coords, ends, side="right"), 0, n_markers - 1)

        for gene, start, end, lo, hi in zip(gene_ids, starts, ends, first, last):
            window = [
                (chrom_coords[i], d)
                for i in range(lo, hi + 1)
                if (d := founder_dosages(chrom_probs[i])) is not None
            ]
            if not window:
                continue
            positions = np.array([p for p, _ in window])
            dosages = np.array([d for _, d in window])
            pair, confidence = call_gene(
                dosages, positions, (start + end) / 2.0, args.hom_dosage_threshold
            )
            diplotype = "".join(haplotypes[i] for i in pair)
            calls.append((gene, diplotype, confidence, len(window)))

    if not calls:
        raise SystemExit("No genes could be genotyped; check the chromosome naming")

    with open(args.out, "w") as f:
        f.write("#Gene_ID\tDiplotype\n")
        for gene, diplotype, _, _ in calls:
            f.write(f"{gene}\t{diplotype}\n")
    if args.out_confidence:
        with open(args.out_confidence, "w") as f:
            f.write("gene_id\tdiplotype\tconfidence\tn_markers\n")
            for gene, diplotype, confidence, n in calls:
                f.write(f"{gene}\t{diplotype}\t{confidence:.4f}\t{n}\n")

    label = args.sample or args.alleleprobs
    confidences = np.array([c for _, _, c, _ in calls])
    print(
        f"{label}: called {len(calls)} genes "
        f"(mean confidence {confidences.mean():.3f}, "
        f"{int((confidences < args.min_call_prob).sum())} below {args.min_call_prob}, "
        f"{int((confidences < 0.99).sum())} below 0.99)",
        file=sys.stderr,
    )
    if known_genes is not None:
        not_in_gtf = known_genes - found_genes
        if not_in_gtf:
            print(
                f"WARNING: {label}: {len(not_in_gtf)} genes of the gene-to-transcript "
                "file are not in the GTF and are omitted",
                file=sys.stderr,
            )
    if n_missing_chrom:
        # Genes on chromosomes with no genotype data (typically Y and MT) are
        # left out; gbrs quantify masks them out of the diploid quantification.
        print(
            f"{label}: {n_missing_chrom} genes on chromosome(s) "
            f"{', '.join(skipped_chroms)} have no genotype data and are omitted",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
