import argparse


def main(args):
    import pathlib
    import polars as pl
    import polars_bio as pb
    import util.compat_classes
    from util.compressed_emase import load_compressed_emase

    tx_annot = (
        pb.scan_gtf(
            args.gtf,
            attr_fields=["gene_id", "transcript_id"],
        )
        .filter(pl.col("type") == "transcript")
        .collect()
    )

    print("Loading data")
    all_compatibility_classes = {}
    for path in map(pathlib.Path, args.h5):
        mouse_id = path.name.removesuffix(".compressed.h5")
        if mouse_id in all_compatibility_classes:
            raise ValueError(f"Mouse {mouse_id} given more than once")
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
    parser.add_argument("--outfile", help="path to write output parquet file")
    parser.add_argument(
        "h5",
        nargs="+",
        help="{mouse_id}.compressed.h5 files to gather (exactly the mice to include)",
    )

    args = parser.parse_args()

    main(args)
