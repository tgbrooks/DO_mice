"""Build a tissue's sample table from a GEO series, for the DO mouse pipeline.

Each tissue in this pipeline is one GEO series of RNA-seq from the same DO
mouse cohort. This script resolves the series into the table the Snakemake
workflow reads, `samples/{tissue}.tsv`:

    mouse_id  geo_sample  experiment  run  layout  sex  generation

It works in two steps, both needing internet access (so run it before
Snakemake, which may be offline on a cluster node):

1. Download the GEO family SOFT file for the series and parse out, per sample,
   the title (which carries the mouse ID, e.g. `DO021`), the SRA experiment
   accession, and any sex / DO generation characteristics.
2. Query the ENA portal API once for the whole SRA study to get each
   experiment's run accession(s) and library layout.

Example:

    python3 scripts/setup_geo_samples.py --tissue Adipose --geo GSE266549

If the GEO lookup is unavailable, pass `--srp SRP...` to skip step 1 and take
mouse IDs from the ENA sample titles instead.
"""

import argparse
import csv
import gzip
import io
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

GEO_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{series}/soft/{series}_family.soft.gz"
)
ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
SRA_STUDY_RE = re.compile(r"\b([SED]RP[0-9]+)\b")
SRA_EXPERIMENT_RE = re.compile(r"\b([SED]RX[0-9]+)\b")


def fetch(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    """GET a URL, retrying with backoff on network errors."""
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - report and retry network errors
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def geo_soft(series: str) -> str:
    """Download and decompress a GEO series' family SOFT file."""
    url = GEO_SOFT_URL.format(prefix=series[:-3], series=series)
    print(f"Fetching {url}", file=sys.stderr)
    raw = fetch(url)
    return gzip.decompress(raw).decode("utf-8", errors="replace")


def parse_soft(text: str, mouse_pattern: re.Pattern) -> tuple[str | None, list[dict]]:
    """Pull the SRA study accession and per-sample metadata out of a SOFT file."""
    study = None
    samples: list[dict] = []
    current: dict | None = None

    for line in io.StringIO(text):
        line = line.rstrip("\n")
        if line.startswith("^SERIES"):
            current = None
        elif line.startswith("^SAMPLE"):
            current = {
                "geo_sample": line.split("=", 1)[1].strip(),
                "title": "",
                "experiment": "",
                "sex": "",
                "generation": "",
            }
            samples.append(current)
        elif line.startswith("!Series_relation") and "SRA" in line:
            match = SRA_STUDY_RE.search(line)
            if match and study is None:
                study = match.group(1)
        elif current is None:
            continue
        elif line.startswith("!Sample_title"):
            current["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_relation") and "SRA" in line:
            match = SRA_EXPERIMENT_RE.search(line)
            if match:
                current["experiment"] = match.group(1)
        elif line.startswith("!Sample_characteristics"):
            value = line.split("=", 1)[1].strip()
            key, _, val = value.partition(":")
            key, val = key.strip().lower(), val.strip()
            if "sex" in key or "gender" in key:
                current["sex"] = val
            elif "generation" in key:
                current["generation"] = val

    for sample in samples:
        match = mouse_pattern.search(sample["title"])
        sample["mouse_id"] = match.group(0) if match else ""
    return study, samples


def ena_runs(accession: str) -> list[dict]:
    """Runs of an SRA study (or experiment) with their layout, from ENA."""
    query = urllib.parse.urlencode(
        {
            "accession": accession,
            "result": "read_run",
            "fields": ",".join(
                [
                    "run_accession",
                    "experiment_accession",
                    "library_layout",
                    "sample_title",
                    "sample_alias",
                ]
            ),
            "format": "tsv",
        }
    )
    text = fetch(f"{ENA_FILEREPORT}?{query}").decode()
    rows = []
    lines = text.splitlines()
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        rows.append(dict(zip(header, fields)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tissue", required=True, help="Tissue name, e.g. Adipose")
    parser.add_argument("--geo", help="GEO series accession, e.g. GSE266549")
    parser.add_argument("--srp", help="SRA study accession; skips the GEO lookup")
    parser.add_argument(
        "--mouse-id-pattern",
        default="DO[0-9]+",
        help="Regex matching the mouse ID within the sample title (default: DO[0-9]+)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Keep only the first N mice (sorted by ID). Handy for a first run.",
    )
    parser.add_argument("--out", help="Output TSV (default: samples/{tissue}.tsv)")
    args = parser.parse_args()

    if not args.geo and not args.srp:
        parser.error("give --geo and/or --srp")

    mouse_pattern = re.compile(args.mouse_id_pattern)
    study = args.srp
    by_experiment: dict[str, dict] = {}

    if args.geo:
        study_from_geo, samples = parse_soft(geo_soft(args.geo), mouse_pattern)
        study = study or study_from_geo
        for sample in samples:
            if sample["experiment"]:
                by_experiment[sample["experiment"]] = sample
        no_id = [s["geo_sample"] for s in samples if not s["mouse_id"]]
        if no_id:
            print(
                f"WARNING: no mouse ID matching /{args.mouse_id_pattern}/ in the title "
                f"of {len(no_id)} sample(s), e.g. {', '.join(no_id[:5])}",
                file=sys.stderr,
            )
        print(f"{args.geo}: {len(samples)} GEO samples, SRA study {study}", file=sys.stderr)

    if not study:
        raise SystemExit(
            f"Could not find the SRA study for {args.geo}. Pass it with --srp."
        )

    runs = ena_runs(study)
    if not runs:
        raise SystemExit(f"ENA returned no runs for {study}")
    print(f"{study}: {len(runs)} runs", file=sys.stderr)

    rows: list[dict] = []
    unmatched = 0
    for run in runs:
        experiment = run.get("experiment_accession", "")
        sample = by_experiment.get(experiment)
        if sample is not None:
            mouse_id = sample["mouse_id"]
        else:
            # No GEO metadata (or --srp only): fall back to the ENA titles.
            text = f"{run.get('sample_title', '')} {run.get('sample_alias', '')}"
            match = mouse_pattern.search(text)
            mouse_id = match.group(0) if match else ""
        if not mouse_id:
            unmatched += 1
            continue
        rows.append(
            {
                "mouse_id": mouse_id,
                "geo_sample": sample["geo_sample"] if sample else "",
                "experiment": experiment,
                "run": run.get("run_accession", ""),
                "layout": (run.get("library_layout", "") or "SINGLE").upper(),
                "sex": sample["sex"] if sample else "",
                "generation": sample["generation"] if sample else "",
            }
        )
    if unmatched:
        print(f"WARNING: {unmatched} runs had no identifiable mouse ID and were skipped",
              file=sys.stderr)

    if args.limit:
        keep = sorted({row["mouse_id"] for row in rows})[: args.limit]
        rows = [row for row in rows if row["mouse_id"] in keep]

    out_path = Path(args.out) if args.out else Path("samples") / f"{args.tissue}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mouse_id", "geo_sample", "experiment", "run", "layout", "sex", "generation"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["mouse_id"], r["run"])))

    n_mice = len({row["mouse_id"] for row in rows})
    print(f"{args.tissue}: {n_mice} mice, {len(rows)} runs -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
