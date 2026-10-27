"""The DeepSeek-V4.1 THD pieces, each against the same work done one sample at a time.

The whole attention stack, at CP=1 and under all-gather CP, is in test_dsv41_thd_packing.py.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout  # noqa: E402
from miles_plugins.models.deepseek_v4_1.engram import DeepSeekV41Engram, compute_hash_multipliers  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.indexer import indexer_select, indexer_select_segments  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.thd import compressor_rows_thd, segment_spans  # noqa: E402

HEADS, DIM, TOPK, BLOCK = 32, 128, 64, 8
SEGMENTS = [1100, 37, 1, 2401, 75]  # odd lengths; one has no group at ratio 2


def _layout(lens, *, global_start=0):
    cu = [0] + torch.tensor(lens).cumsum(0).tolist()
    return ThdLayout(
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        seq_lens=tuple(lens),
        global_start=global_start,
        max_seqlen=max(lens),
    )


def _positions(lens):
    return torch.cat([torch.arange(n, device="cuda") for n in lens])


@pytest.mark.parametrize("cp_size", [1, 2, 3, 5])
@pytest.mark.parametrize("ratio", [1, 2])
def test_segment_spans_cover_each_rank_once(cp_size, ratio):
    total = sum(SEGMENTS) + (-sum(SEGMENTS)) % cp_size
    lens = SEGMENTS + ([total - sum(SEGMENTS)] if total > sum(SEGMENTS) else [])
    local = total // cp_size
    starts = [0] + torch.tensor(lens).cumsum(0).tolist()
    key_starts = [0] + torch.tensor([n // ratio for n in lens]).cumsum(0).tolist()
    seen = []
    for rank in range(cp_size):
        offset = rank * local
        for span in segment_spans(_layout(lens, global_start=offset), ratio=ratio, seqlen_local=local):
            first, last = offset + span.row_start, offset + span.row_end
            seg = max(i for i in range(len(lens)) if starts[i] <= first)
            assert last <= starts[seg + 1], "a span crosses a segment boundary"
            assert (span.key_start, span.key_end) == (key_starts[seg], key_starts[seg + 1])
            seen.extend(range(first, last))
    assert seen == list(range(total))


@pytest.mark.parametrize("topk_backend", ["canonical", "torch"])
@pytest.mark.parametrize("candidates", ["none", "source", "user"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_indexer_segments_match_each_segment_alone(topk_backend, candidates, ratio):
    torch.manual_seed(0)
    total = sum(SEGMENTS)
    n_kv = [n // ratio for n in SEGMENTS]
    q = torch.randn(1, total, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(1, sum(n_kv), DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, total, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    compress_lens = (_positions(SEGMENTS) + 1) // ratio
    spans = segment_spans(_layout(SEGMENTS), ratio=ratio, seqlen_local=total)
    width = max(n_kv)
    user_mask = torch.rand(1, total, width, device="cuda") < 0.3 if candidates == "user" else None
    options = dict(
        is_candidate_source=candidates == "source",
        uses_candidates=candidates == "user",
        candidate_topk_blocks=4,
        candidate_block_size=BLOCK,
        topk=TOPK,
        topk_fn=get_dsa_topk_fn(topk_backend),
        canonical_topk=topk_backend == "canonical",
        allow_deep_select=False,
        query_chunk=512,
    )
    idx, cand = indexer_select_segments(q, index_k, weights, compress_lens, user_mask, spans, **options)
    assert idx.shape == (1, total, TOPK)
    for span in spans:
        rows, keys = slice(span.row_start, span.row_end), slice(span.key_start, span.key_end)
        n = span.key_end - span.key_start
        if n == 0:
            assert bool((idx[:, rows] == -1).all())
            continue
        alone_idx, alone_cand = indexer_select(
            q[:, rows],
            index_k[:, keys],
            weights[:, rows],
            compress_lens[rows],
            user_mask[:, rows, :n] if user_mask is not None else None,
            **options,
        )
        assert torch.equal(idx[:, rows], torch.where(alone_idx >= 0, alone_idx + span.key_start, -1))
        picked = idx[:, rows]
        assert bool(((picked == -1) | ((picked >= span.key_start) & (picked < span.key_end))).all())
        if candidates == "source":
            assert torch.equal(cand[:, rows, :n], alone_cand)
            assert not cand[:, rows, n:].any()


def test_compressor_rows_group_within_segments():
    torch.manual_seed(0)
    total = sum(SEGMENTS)
    hidden = torch.randn(total, 1, 16, device="cuda")
    rows = compressor_rows_thd(hidden, _layout(SEGMENTS), ratio=2, cp_group=None)
    want, positions, start = [], [], 0
    for n in SEGMENTS:
        want.append(hidden[start : start + n // 2 * 2])
        positions.append(torch.arange(n // 2, device="cuda") * 2)
        start += n
    assert torch.equal(rows.hidden, torch.cat(want))
    assert torch.equal(rows.positions, torch.cat(positions))
    assert rows.seq_rows is None


def test_engram_hash_restarts_at_segments():
    torch.manual_seed(0)
    max_ngram = 4
    vocab = 1000
    primes = torch.tensor([[1009, 1013], [1019, 1021], [1031, 1033]], device="cuda")
    engram = SimpleNamespace(
        token_map=torch.randperm(vocab, device="cuda"),
        multipliers=compute_hash_multipliers([1], max_ngram, vocab)[0].cuda(),
        primes=primes,
        offsets=torch.tensor([0, 1009, 2022, 3041, 4062, 5093], device="cuda"),
        pad_id=7,
        max_ngram_size=max_ngram,
    )
    lens = [5, 1, 2, 9, 3]
    ids = torch.randint(0, vocab, (1, sum(lens)), device="cuda")
    packed = DeepSeekV41Engram.hash_ids(engram, ids, _positions(lens))
    alone = [DeepSeekV41Engram.hash_ids(engram, seg) for seg in ids.split(lens, dim=1)]
    assert torch.equal(packed, torch.cat(alone, dim=1))
    # Without the positions every segment after the first hashes its predecessor's tail.
    assert not torch.equal(DeepSeekV41Engram.hash_ids(engram, ids), packed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
