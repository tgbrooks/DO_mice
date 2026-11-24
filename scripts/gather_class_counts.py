import argparse


def main(args):
    import pathlib
    import polars as pl
    import polars_bio as pb
    import util.compat_classes
    from util.compressed_emase import load_compressed_emase

    gene_annot = (
        pb.scan_gtf(
            args.gtf,
            attr_fields=["gene_id", "gene_name", "gene_biotype"],
        )
        .filter(pl.col("type") == "gene")
        .collect()
    )

    tx_annot = (
        pb.scan_gtf(
            args.gtf,
            attr_fields=["gene_id", "transcript_id"],
        )
        .filter(pl.col("type") == "transcript")
        .collect()
    )

    SKIP_IDS = args.skip_ids.strip().split(",")

    print("Loading data")
    all_compatibility_classes = {}
    input_dir = pathlib.Path(args.input_dir)
    for path in input_dir.glob("*.compressed.h5"):
        mouse_id, _, _ = path.name.split(".")
        if mouse_id in SKIP_IDS:
            continue
        all_compatibility_classes[mouse_id] = load_compressed_emase(str(path))

    print("Processing data")
    df = util.compat_classes.get_all_gene_class_counts(
        tx_annot,
        all_compatibility_classes,
    )
    print("Writing output")
    df.write_parquet(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "gather gene-level data per compatibility class uniformized across all samples"
    )
    parser.add_argument("--gtf", help="gtf file path of annotation")
    parser.add_argument(
        "--input_dir",
        help="directory containing {mouse_id}.compressed.h5 files to gather",
    )
    parser.add_argument(
        "--skip_ids", help="comma-delimited list of mouse ids to skip (e.g., outliers)"
    )
    parser.add_argument("--outfile", help="path to write output parquet file")

    args = parser.parse_args()

    main(args)
