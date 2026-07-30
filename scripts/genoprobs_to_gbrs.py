"""Convert one mouse's founder haplotype probabilities into GBRS genotype calls.

`gbrs quantify -G` takes a two-column file of per-gene diplotype calls:

    #Gene_ID	Diplotype
    ENSMUSG00000000001	DF
    ENSMUSG00000000003	CC

This script produces that file from array-based founder probabilities (exported
from the genoprobs .RData file by scripts/export_genoprobs.R), so that RNA-seq
can be quantified against each mouse's known genome instead of one reconstructed
from the expression data.

Each gene is assigned the founder probabilities of its nearest genotyped marker.
The probabilities are converted to founder dosages (summing to 2) and called as
homozygous when the top founder's dosage reaches --hom-dosage-threshold, and as
the heterozygous combination of the top two founders otherwise.

Positions are matched in whichever coordinate the GBRS gene position file uses
(cM or bp, detected automatically). Marker coordinates come, in order of
preference, from the GBRS genome grid (matching marker names), from the marker
name itself when it encodes a position (e.g. `1_3000000`), or from the marker
table written by export_genoprobs.R. cM positions for markers that are not in
the grid are interpolated from the grid's own bp/cM columns.

Only numpy is required, so this runs inside the GBRS container.
"""

import argparse
import gzip
import re
import sys
from collections import OrderedDict

import numpy as np

MARKER_POS_RE = re.compile(r"^(?:chr)?([0-9]+|[XYMxym]|MT|mt)[_:.-]([0-9]+)$")


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


def read_grid(path: str):
    """Read the GBRS genome grid.

    Returns (by_name, by_chrom) where by_name maps a grid marker name to
    (chrom, bp, cM) and by_chrom maps a chromosome to sorted (bp, cM) arrays for
    interpolating positions of markers that are not in the grid.
    """
    by_name: dict[str, tuple[str, float, float]] = {}
    rows: dict[str, list[tuple[float, float]]] = {}
    with open(path) as f:
        first = f.readline()
        if not first.lower().startswith(("marker", "#")):
            f.seek(0)
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            name = fields[0]
            chrom = normalize_chrom(fields[1])
            cm = float(fields[3])
            # Columns are marker, chr, pos, cM[, bp]; when a separate bp column
            # is absent, `pos` is the base-pair position.
            bp = float(fields[4]) if len(fields) > 4 else float(fields[2])
            by_name[name] = (chrom, bp, cm)
            rows.setdefault(chrom, []).append((bp, cm))
    by_chrom = {}
    for chrom, pairs in rows.items():
        arr = np.array(sorted(pairs), dtype=float)
        by_chrom[chrom] = (arr[:, 0], arr[:, 1])
    return by_name, by_chrom


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


def read_gene_positions(path: str):
    """Read the GBRS gene position NPZ: chromosome -> (gene IDs, positions)."""
    data = np.load(path, allow_pickle=False)
    genes = OrderedDict()
    for chrom in data.files:
        entries = data[chrom]
        ids = []
        positions = []
        for gene, pos in entries:
            ids.append(gene.decode() if isinstance(gene, bytes) else str(gene))
            positions.append(float(pos.decode() if isinstance(pos, bytes) else pos))
        genes[normalize_chrom(chrom)] = (np.array(ids), np.array(positions, dtype=float))
    return genes


def detect_gene_units(gene_positions) -> str:
    """Guess whether gene positions are in bp or cM.

    Chromosome-scale bp positions run to 10^8, genetic positions to ~10^2, so
    the largest position separates the two unambiguously.
    """
    largest = max(
        (positions.max() for _, positions in gene_positions.values() if positions.size),
        default=0.0,
    )
    return "bp" if largest > 1e6 else "cM"


def marker_coordinates(
    markers: np.ndarray,
    chroms: np.ndarray,
    grid_by_name: dict,
    grid_by_chrom: dict,
    marker_table: dict,
    marker_units: str,
    units: str,
) -> np.ndarray:
    """Coordinate of every marker in `units` ('bp' or 'cM'); NaN when unknown."""
    coords = np.full(len(markers), np.nan)
    bp = np.full(len(markers), np.nan)
    cm = np.full(len(markers), np.nan)

    for i, (name, chrom) in enumerate(zip(markers, chroms)):
        if name in grid_by_name:
            _, marker_bp, marker_cm = grid_by_name[name]
            bp[i], cm[i] = marker_bp, marker_cm
            continue
        match = MARKER_POS_RE.match(name)
        if match:
            bp[i] = float(match.group(2))
            continue
        if name in marker_table:
            pos = marker_table[name]
            if marker_units == "bp":
                bp[i] = pos
            elif marker_units == "Mbp":
                bp[i] = pos * 1e6
            elif marker_units == "cM":
                cm[i] = pos
            elif marker_units == "auto":
                if pos > 1e6:
                    bp[i] = pos
                else:
                    raise SystemExit(
                        "Cannot tell whether the marker positions in the marker table "
                        f"are Mbp or cM (largest value {pos}). Set genotypes: "
                        "marker_units: to 'bp', 'Mbp', or 'cM' in config.yaml."
                    )

    if units == "bp":
        coords = bp
    else:
        # Fill in cM for markers that only have bp, by interpolating the grid.
        coords = cm
        need = np.isnan(coords) & ~np.isnan(bp)
        for chrom in np.unique(chroms[need]):
            if chrom not in grid_by_chrom:
                continue
            grid_bp, grid_cm = grid_by_chrom[chrom]
            sel = need & (chroms == chrom)
            coords[sel] = np.interp(bp[sel], grid_bp, grid_cm)
    return coords


def call_diplotype(probs: np.ndarray, haplotypes: list[str], hom_threshold: float):
    """Call a diplotype from one marker's founder probabilities.

    Returns (diplotype, probability of the call). Founders are ordered as in
    `haplotypes`, matching the diplotype naming GBRS itself uses.
    """
    total = probs.sum()
    if total <= 0:
        return None, 0.0
    fractions = probs / total
    dosage = 2.0 * fractions
    order = np.argsort(dosage)[::-1]
    top = int(order[0])
    if dosage[top] >= hom_threshold:
        pair = (top, top)
        confidence = float(fractions[top])
    else:
        second = int(order[1])
        pair = tuple(sorted((top, second)))
        confidence = float(fractions[top] + fractions[second])
    return "".join(haplotypes[i] for i in pair), confidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alleleprobs", required=True, help="Per-mouse founder probability TSV")
    parser.add_argument("--grid", required=True, help="GBRS genome grid TSV")
    parser.add_argument("--gene-pos", required=True, help="GBRS gene position NPZ")
    parser.add_argument("--markers", help="Marker position table from export_genoprobs.R")
    parser.add_argument(
        "--gene2transcripts",
        help="EMASE gene-to-transcript file; restricts calls to the genes GBRS knows",
    )
    parser.add_argument("--out", required=True, help="Output genotype calls TSV")
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
        help="Report how many calls fall below this probability",
    )
    parser.add_argument(
        "--marker-units",
        default="auto",
        choices=["auto", "bp", "Mbp", "cM"],
        help="Units of positions in the --markers table (only used as a fallback)",
    )
    parser.add_argument(
        "--gene-pos-units",
        default="auto",
        choices=["auto", "bp", "cM"],
        help="Units of the GBRS gene positions (detected by default)",
    )
    args = parser.parse_args()

    haplotypes = args.haplotypes.split(",")
    markers, marker_chroms, probs = read_alleleprobs(args.alleleprobs, haplotypes)
    grid_by_name, grid_by_chrom = read_grid(args.grid)
    marker_table = read_marker_table(args.markers)
    gene_positions = read_gene_positions(args.gene_pos)
    known_genes = read_known_genes(args.gene2transcripts)

    units = args.gene_pos_units
    if units == "auto":
        units = detect_gene_units(gene_positions)
    print(f"Matching genes to markers in {units}", file=sys.stderr)

    coords = marker_coordinates(
        markers,
        marker_chroms,
        grid_by_name,
        grid_by_chrom,
        marker_table,
        args.marker_units,
        units,
    )
    known = ~np.isnan(coords)
    if not known.any():
        raise SystemExit(
            "None of the marker positions could be resolved. The marker names in the "
            "genotype file match neither the GBRS genome grid nor a `chr_position` "
            "pattern, and no usable marker table was given. Supply positions via the "
            "map object in the .RData file (see scripts/export_genoprobs.R)."
        )
    if not known.all():
        print(
            f"WARNING: {int((~known).sum())} of {len(markers)} markers have no "
            "resolvable position and are ignored",
            file=sys.stderr,
        )

    calls: list[tuple[str, str]] = []
    n_low = 0
    n_missing_chrom = 0
    n_unknown_gene = 0
    confidences: list[float] = []
    skipped_chroms: list[str] = []

    for chrom, (gene_ids, gene_pos) in gene_positions.items():
        on_chrom = known & (marker_chroms == chrom)
        if not on_chrom.any():
            n_missing_chrom += len(gene_ids)
            skipped_chroms.append(chrom)
            continue
        chrom_coords = coords[on_chrom]
        chrom_probs = probs[on_chrom]
        order = np.argsort(chrom_coords)
        chrom_coords = chrom_coords[order]
        chrom_probs = chrom_probs[order]

        # Nearest marker for each gene.
        right = np.searchsorted(chrom_coords, gene_pos)
        right = np.clip(right, 1, len(chrom_coords) - 1) if len(chrom_coords) > 1 else np.zeros_like(right)
        left = np.maximum(right - 1, 0)
        pick_left = np.abs(gene_pos - chrom_coords[left]) <= np.abs(chrom_coords[right] - gene_pos)
        nearest = np.where(pick_left, left, right)

        for gene, idx in zip(gene_ids, nearest):
            if known_genes is not None and gene not in known_genes:
                n_unknown_gene += 1
                continue
            diplotype, confidence = call_diplotype(
                chrom_probs[idx], haplotypes, args.hom_dosage_threshold
            )
            if diplotype is None:
                continue
            calls.append((gene, diplotype))
            confidences.append(confidence)
            if confidence < args.min_call_prob:
                n_low += 1

    if not calls:
        raise SystemExit("No genes could be genotyped; check the chromosome naming")

    with open(args.out, "w") as f:
        f.write("#Gene_ID\tDiplotype\n")
        for gene, diplotype in calls:
            f.write(f"{gene}\t{diplotype}\n")

    label = args.sample or args.alleleprobs
    print(
        f"{label}: called {len(calls)} genes "
        f"(mean call probability {np.mean(confidences):.3f}, "
        f"{n_low} below {args.min_call_prob})",
        file=sys.stderr,
    )
    if n_unknown_gene:
        print(
            f"{label}: {n_unknown_gene} genes are in the gene position file but not "
            "in the gene-to-transcript file, and are omitted",
            file=sys.stderr,
        )
    if n_missing_chrom:
        # Genes on chromosomes with no genotype data (typically Y and MT) are
        # left out; gbrs quantify masks them out of the diploid quantification.
        print(
            f"{label}: {n_missing_chrom} genes on chromosome(s) "
            f"{', '.join(skipped_chroms)} have no genotype data and are omitted; "
            "they get zero expression in the diploid quantification",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
