"""THD (packed variable-length) layout for the DeepSeek-V4.1 attention.

The window, the rotary positions and the CP compaction of a compressed group that straddles the
split come from the DeepSeek-V4 helpers in ``deepseek_v4/ops/thd_utils.py``. What V4.1 adds:

- A kv_source layer's compressed latent and index keys are reused by the later layers of its
  ratio, so they are kept sequence-major (segment after segment) once gathered, and every reader
  addresses them the same way.
- The indexer scores each segment against that segment's keys only (``SegmentSpan``), the way it
  scores a sample that runs alone, which also keeps the candidate blocks aligned to the segment's
  first key.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    CompressorInputCompact,
    ThdLayout,
    batch_of_row,
    compact_group_capacity,
    compressed_cu_seqlens,
    compressed_rank_layout,
    exchange_cp_boundary_hidden,
    get_q_positions_thd,
)


@dataclass(frozen=True)
class SegmentSpan:
    """One segment's share of this rank: its local rows and its sequence-major compressed keys."""

    row_start: int
    row_end: int
    key_start: int
    key_end: int


def segment_spans(thd: ThdLayout, *, ratio: int, seqlen_local: int) -> list[SegmentSpan]:
    """The segments this rank holds rows of, read off the host lengths so nothing syncs."""
    spans, start, key_start = [], 0, 0
    lo, hi = thd.global_start, thd.global_start + seqlen_local
    for n in thd.seq_lens:
        first, last = max(start, lo), min(start + n, hi)
        if first < last:
            spans.append(SegmentSpan(first - lo, last - lo, key_start, key_start + n // ratio))
        start += n
        key_start += n // ratio
    return spans


@dataclass
class CompressorRows:
    """What the compressor reads under THD and where its output rows end up.

    ``hidden`` holds each group's tokens back to back, the layout the compressor groups by.
    ``positions`` is the in-segment position of each group's first token, for the rotary.
    ``seq_rows`` reorders the CP all-gather of the output into sequence-major rows; None when the
    gather is already in that order.
    """

    hidden: Tensor
    positions: Tensor
    seq_rows: Tensor | None


def compressor_rows_thd(hidden: Tensor, thd: ThdLayout, *, ratio: int, cp_group) -> CompressorRows:
    """Group this rank's packed rows ``[s, b, d]`` within their segments.

    Ratio 1 groups nothing. Otherwise a segment's tail shorter than ``ratio`` gets no group, as
    in the BSHD path, and under CP a group that straddles the split is built by the rank holding
    its last token, from rows its left neighbour sends.
    """
    seqlen = hidden.size(0)
    if ratio == 1:
        return CompressorRows(hidden, get_q_positions_thd(thd.cu_seqlens, seqlen, thd.global_start), None)
    cp_size = cp_group.size() if cp_group is not None else 1
    n_groups = sum(n // ratio for n in thd.seq_lens)
    if cp_size == 1:
        cu_compressed = compressed_cu_seqlens(thd.cu_seqlens, ratio)
        seg = batch_of_row(cu_compressed, n_groups)
        group_ids = torch.arange(n_groups, device=hidden.device) - cu_compressed[seg]
        heads = thd.cu_seqlens[seg] + group_ids * ratio
        rows = (heads.unsqueeze(1) + torch.arange(ratio, device=hidden.device)).flatten()
        return CompressorRows(hidden.index_select(0, rows.long()), group_ids * ratio, None)
    boundary = exchange_cp_boundary_hidden(hidden, ratio=ratio, cp_group=cp_group)
    c_cap = compact_group_capacity(seqlen, ratio)
    compact, group_ids = CompressorInputCompact.apply(hidden, boundary, thd.cu_seqlens, thd.global_start, ratio, c_cap)
    seq_rows = compressed_rank_layout(
        thd.cu_seqlens,
        compressed_cu_seqlens(thd.cu_seqlens, ratio),
        l_local=seqlen,
        cp_size=cp_size,
        ratio=ratio,
        c_cap=c_cap,
    )
    # Rows past the last group are -1; dropping them leaves the same row count as without CP.
    return CompressorRows(compact, group_ids.clamp(min=0) * ratio, seq_rows[:n_groups].long())
