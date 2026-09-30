"""
Pair up the two ends of each read from separate per-end BAM files.

GBRS aligns each end on its own, so the ends of a read are in two BAMs. Both
list reads in roughly FASTQ order, but not exactly: bowtie's threads reorder
reads locally, and one file can hold reads the other lacks. `paired_reads`
matches ends by name while streaming both files, keeping in memory only the
ends still waiting for their mate, so memory depends on how far the two
orderings drift apart rather than on the size of the files.
"""

import itertools
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pysam


def read_groups(bam: pysam.AlignmentFile) -> Iterator[tuple[str, list[pysam.AlignedSegment]]]:
    """(read name, its alignments) for each read in file order.

    Assumes a read's alignments are consecutive, as bowtie writes them.
    """
    for name, alignments in itertools.groupby(
        bam.fetch(until_eof=True), key=lambda a: a.query_name
    ):
        yield name, list(alignments)


@dataclass
class PairingStats:
    reads: tuple[int, int] = (0, 0)
    pairs: int = 0
    # Ends whose mate was never found within the window (or at all)
    orphans: int = 0
    # Largest number of ends held waiting for their mate at once
    max_pending: int = 0


def paired_reads(
    r1_bam: pysam.AlignmentFile,
    r2_bam: pysam.AlignmentFile,
    summarize: Callable[[list[pysam.AlignedSegment]], Any],
    window: int = 1_000_000,
    max_reads: int | None = None,
    stats: PairingStats | None = None,
) -> Iterator[tuple[str, Any, Any]]:
    """Yield (read name, R1 summary, R2 summary) for each read in both files.

    `summarize` reduces one end's alignments to whatever the caller needs, and
    only that is held while waiting for the other end, so keep it small.

    The files are read alternately: each keeps reading while its reads are ones
    the other file has already passed, and hands over once it reads one the
    other hasn't reached. An end still unmatched once its own file has moved
    `window` reads past it is dropped as an orphan. Reading stops after
    `max_reads` reads of either file.
    """
    if stats is None:
        stats = PairingStats()
    streams = [read_groups(r1_bam), read_groups(r2_bam)]
    pending: list[OrderedDict[str, tuple[int, Any]]] = [OrderedDict(), OrderedDict()]
    count = [0, 0]
    done = [False, False]
    side = 0
    while not (done[0] and done[1]):
        if done[side]:
            side = 1 - side
        if max_reads is not None and count[side] >= max_reads:
            break
        try:
            name, alignments = next(streams[side])
        except StopIteration:
            done[side] = True
            side = 1 - side
            continue
        count[side] += 1
        summary = summarize(alignments)

        other = pending[1 - side]
        if name in other:
            _, other_summary = other.pop(name)
            stats.pairs += 1
            if side == 0:
                yield name, summary, other_summary
            else:
                yield name, other_summary, summary
            continue

        mine = pending[side]
        mine[name] = (count[side], summary)
        while mine and next(iter(mine.values()))[0] < count[side] - window:
            mine.popitem(last=False)
            stats.orphans += 1
        stats.max_pending = max(stats.max_pending, len(mine) + len(other))
        if not done[1 - side]:
            side = 1 - side

    stats.orphans += len(pending[0]) + len(pending[1])
    stats.reads = (count[0], count[1])
