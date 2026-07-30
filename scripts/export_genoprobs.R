#!/usr/bin/env Rscript

## Extract founder haplotype probabilities for individual mice from an .RData
## file, so they can be converted into GBRS genotype calls without R.
##
## The .RData file is expected to hold a list (default name `genoprobs`) with
## one entry per chromosome ("1".."19", "X"), each a 3-D array of mice x 8
## founders x markers. The margins are identified from the dimnames rather than
## assumed, so a different ordering still works.
##
## Two modes:
##
##   # just list the mouse IDs in the file
##   Rscript export_genoprobs.R --rdata geno/genoprobs.RData --list-mice out.txt
##
##   # export per-mouse founder probabilities
##   Rscript export_genoprobs.R --rdata geno/genoprobs.RData \
##       --mice DO021,DO022 --outdir geno/alleleprobs \
##       --markers geno/markers.tsv --founder-order geno/founder_order.txt
##
## Each per-mouse file (`{outdir}/{mouse}.tsv.gz`) has columns:
##   marker  chr  A  B  C  D  E  F  G  H
## where the founder columns are the haplotype letters in the order they appear
## in the genoprobs array (the original founder labels are recorded in the
## --founder-order file so the assignment can be checked).
##
## `--markers` receives marker positions if the .RData file also contains a map
## object (a list of named numeric vectors per chromosome, e.g. `map`, `gmap`,
## or `pmap`). It is written empty (header only) when no such object is found,
## in which case marker positions are resolved against the GBRS genome grid
## instead (see scripts/genoprobs_to_gbrs.py).

args <- commandArgs(trailingOnly = TRUE)

get_opt <- function(name, default = NULL) {
    hit <- which(args == paste0("--", name))
    if (length(hit) == 0) {
        return(default)
    }
    if (hit[1] == length(args)) {
        stop(sprintf("--%s needs a value", name))
    }
    args[hit[1] + 1]
}

rdata_file <- get_opt("rdata")
object_name <- get_opt("object", "genoprobs")
list_mice_file <- get_opt("list-mice")
mice_arg <- get_opt("mice")
outdir <- get_opt("outdir")
markers_file <- get_opt("markers")
founder_order_file <- get_opt("founder-order")
haplotypes <- strsplit(get_opt("haplotypes", "A,B,C,D,E,F,G,H"), ",")[[1]]

if (is.null(rdata_file)) {
    stop("--rdata is required")
}
if (is.null(list_mice_file) && is.null(outdir)) {
    stop("give either --list-mice or --outdir")
}

message("Loading ", rdata_file)
env <- new.env()
loaded <- load(rdata_file, envir = env)
message("Objects in file: ", paste(loaded, collapse = ", "))

if (!(object_name %in% loaded)) {
    stop(sprintf(
        "no object named '%s' in %s (found: %s)",
        object_name, rdata_file, paste(loaded, collapse = ", ")
    ))
}
genoprobs <- get(object_name, envir = env)
if (!is.list(genoprobs)) {
    stop(sprintf("'%s' is not a list of per-chromosome arrays", object_name))
}
chroms <- names(genoprobs)
if (is.null(chroms)) {
    stop(sprintf("'%s' has no chromosome names", object_name))
}
message("Chromosomes: ", paste(chroms, collapse = ", "))

## Identify which array margin is mice, which is founders, and which is markers.
## The mouse margin is the one whose dimnames look like sample IDs; of the two
## remaining margins the founder one has length 8 (or length(haplotypes)).
margins_of <- function(arr, chrom) {
    dims <- dim(arr)
    if (length(dims) != 3) {
        stop(sprintf("genoprobs[['%s']] is not a 3-D array", chrom))
    }
    dn <- dimnames(arr)
    if (is.null(dn)) {
        stop(sprintf("genoprobs[['%s']] has no dimnames", chrom))
    }
    n_hap <- length(haplotypes)
    # Founder margin: length 8 and short labels (founder codes, not marker names).
    founder_dim <- which(dims == n_hap)
    if (length(founder_dim) > 1) {
        # Ambiguous by size alone; prefer the margin whose labels are shortest.
        widths <- sapply(founder_dim, function(i) {
            if (is.null(dn[[i]])) Inf else max(nchar(dn[[i]]))
        })
        founder_dim <- founder_dim[which.min(widths)]
    }
    if (length(founder_dim) != 1) {
        stop(sprintf(
            "genoprobs[['%s']] has no margin of length %d for the founders (dims: %s)",
            chrom, n_hap, paste(dims, collapse = " x ")
        ))
    }
    # Marker margin: the longest of the two remaining margins.
    rest <- setdiff(seq_len(3), founder_dim)
    marker_dim <- rest[which.max(dims[rest])]
    sample_dim <- setdiff(rest, marker_dim)
    list(sample = sample_dim, founder = founder_dim, marker = marker_dim)
}

first_chrom <- chroms[1]
m <- margins_of(genoprobs[[first_chrom]], first_chrom)
mouse_ids <- dimnames(genoprobs[[first_chrom]])[[m$sample]]
if (is.null(mouse_ids)) {
    stop("the mouse margin of the genoprobs arrays has no dimnames")
}
message("Mice in file: ", length(mouse_ids))

if (!is.null(list_mice_file)) {
    dir.create(dirname(list_mice_file), showWarnings = FALSE, recursive = TRUE)
    writeLines(mouse_ids, list_mice_file)
    message("Wrote ", length(mouse_ids), " mouse IDs to ", list_mice_file)
    if (is.null(outdir)) {
        quit(save = "no", status = 0)
    }
}

wanted <- if (is.null(mice_arg) || mice_arg == "") {
    character(0)
} else {
    strsplit(mice_arg, ",")[[1]]
}
missing <- setdiff(wanted, mouse_ids)
if (length(missing) > 0) {
    stop(sprintf(
        "%d requested mice are not in %s: %s",
        length(missing), rdata_file, paste(missing, collapse = ", ")
    ))
}

founder_labels <- dimnames(genoprobs[[first_chrom]])[[m$founder]]
if (is.null(founder_labels)) {
    founder_labels <- haplotypes
}
if (!is.null(founder_order_file)) {
    dir.create(dirname(founder_order_file), showWarnings = FALSE, recursive = TRUE)
    writeLines(
        c(
            "# founder columns written by export_genoprobs.R, in array order",
            "# haplotype\tlabel_in_genoprobs",
            paste(haplotypes, founder_labels, sep = "\t")
        ),
        founder_order_file
    )
}

## Collect per-chromosome marker names, and the founder probabilities of each
## requested mouse, as one matrix per mouse.
marker_names <- list()
marker_chrom <- list()
per_mouse <- setNames(vector("list", length(wanted)), wanted)

for (chrom in chroms) {
    arr <- genoprobs[[chrom]]
    mm <- margins_of(arr, chrom)
    dn <- dimnames(arr)
    ids <- dn[[mm$sample]]
    markers <- dn[[mm$marker]]
    if (is.null(markers)) {
        stop(sprintf(
            "genoprobs[['%s']] has no marker names; marker positions cannot be resolved",
            chrom
        ))
    }
    marker_names[[chrom]] <- markers
    marker_chrom[[chrom]] <- rep(chrom, length(markers))

    # Reorder to markers x founders for every requested mouse.
    perm <- c(mm$sample, mm$marker, mm$founder)
    arr <- aperm(arr, perm)
    for (mouse in wanted) {
        idx <- match(mouse, ids)
        if (is.na(idx)) {
            stop(sprintf("mouse %s is missing from chromosome %s", mouse, chrom))
        }
        block <- arr[idx, , , drop = FALSE]
        dim(block) <- c(length(markers), length(founder_labels))
        per_mouse[[mouse]][[chrom]] <- block
    }
}

all_markers <- unlist(marker_names, use.names = FALSE)
all_chroms <- unlist(marker_chrom, use.names = FALSE)

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
for (mouse in wanted) {
    probs <- do.call(rbind, per_mouse[[mouse]])
    colnames(probs) <- haplotypes
    out <- data.frame(
        marker = all_markers,
        chr = all_chroms,
        probs,
        check.names = FALSE,
        stringsAsFactors = FALSE
    )
    path <- file.path(outdir, paste0(mouse, ".tsv.gz"))
    con <- gzfile(path, "w")
    write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
    close(con)
    message("Wrote ", nrow(out), " markers to ", path)
}

## Marker positions, if the file also ships a map.
write_markers <- function(path, map_obj, source_name) {
    rows <- do.call(rbind, lapply(names(map_obj), function(chrom) {
        positions <- map_obj[[chrom]]
        if (is.null(names(positions))) {
            return(NULL)
        }
        data.frame(
            marker = names(positions),
            chr = chrom,
            pos = as.numeric(positions),
            stringsAsFactors = FALSE
        )
    }))
    dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
    if (is.null(rows)) {
        rows <- data.frame(marker = character(0), chr = character(0), pos = numeric(0))
    }
    write.table(rows, path, sep = "\t", quote = FALSE, row.names = FALSE)
    message(
        "Wrote ", nrow(rows), " marker positions to ", path,
        if (is.null(source_name)) "" else paste0(" (from '", source_name, "')")
    )
}

if (!is.null(markers_file)) {
    map_obj <- NULL
    map_source <- NULL
    for (candidate in c("map", "gmap", "pmap", "grid", "markers")) {
        if (!(candidate %in% loaded)) {
            next
        }
        obj <- get(candidate, envir = env)
        if (is.list(obj) && all(sapply(obj, is.numeric))) {
            map_obj <- obj
            map_source <- candidate
            break
        }
    }
    if (is.null(map_obj)) {
        message(
            "No map object found in ", rdata_file,
            "; writing an empty ", markers_file,
            " (marker positions will be taken from the GBRS genome grid)"
        )
        write_markers(markers_file, list(), NULL)
    } else {
        write_markers(markers_file, map_obj, map_source)
    }
}

message("Done")
