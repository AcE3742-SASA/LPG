"""Geometry-only selection from the existing full main plan; no timing input."""
from collections import Counter, defaultdict
import hashlib
import json
import math

SEED = 20260921
COUNT = 200
ANCHORS = [(v, v, v) for v in (64, 128, 256, 512, 1024, 2048, 4096)] + [
    (64, 4096, 64), (4096, 64, 64), (64, 64, 4096)]


def rank(value):
    return hashlib.sha256(json.dumps([SEED, value], separators=(',', ':')).encode()).hexdigest()


def stratum(shape):
    volume = sum(2 * math.prod(shape) >= 2 ** edge for edge in (24, 28, 32))
    aspect = 'elongated' if max(shape) >= 4 * min(shape) else 'compact'
    alignment = 'aligned8' if all(v % 8 == 0 for v in shape) else 'unaligned8'
    boundary = 'near_power2' if any(abs(v - edge) <= 1 for v in shape
                                   for edge in (64, 128, 256, 512, 1024, 2048, 4096)) else 'other'
    return f'{volume}/{aspect}/{alignment}/{boundary}'


def select(original):
    candidates = {tuple(t['shape']): t['split'] for t in original['tasks']
                  if t['kind'] == 'gemm' and t.get('phase') == 'main' and t['split'] != 'fresh_holdout'}
    groups = defaultdict(list)
    for shape in candidates:
        groups[stratum(shape)].append(shape)
    keys = sorted(groups, key=rank)
    if len(keys) != 32 or not set(ANCHORS) <= set(candidates):
        raise ValueError('Unexpected geometry pool; inspect rather than change selection silently')
    base, remainder = divmod(COUNT, len(keys))
    selected, quotas = [], []
    for index, key in enumerate(keys):
        quota = base + (index < remainder)
        anchors = sorted(s for s in ANCHORS if stratum(s) == key)
        others = sorted((s for s in groups[key] if s not in ANCHORS), key=rank)
        chosen = anchors + others[:quota - len(anchors)]
        if len(anchors) > quota or len(chosen) != quota:
            raise ValueError('Geometry stratum cannot meet its frozen quota')
        quotas.append(dict(stratum=key, available=len(groups[key]), selected=quota))
        selected.extend(dict(shape=list(s), split=candidates[s], stratum=key, anchor=s in ANCHORS)
                        for s in chosen)
    return dict(version=1, seed=SEED, count=COUNT, candidates=len(candidates), strata=quotas,
                selection_rule='10 geometric anchors plus SHA256-ranked geometry strata; no latency/status inputs',
                size_bands_flops=[2**24, 2**28, 2**32], aspect_threshold=4,
                alignment_multiple=8, power2_boundary_distance=1,
                fresh_holdout_excluded=True, previous_legacy_results_already_explored=True,
                purpose='descriptive subset; not an untouched prediction test set', shapes=selected)


def validate(original, saved):
    if saved != select(original):
        raise ValueError('Frozen shape selection differs from the reproducible geometry rule')
    shapes = {tuple(row['shape']) for row in saved['shapes']}
    if len(shapes) != COUNT:
        raise ValueError('Expected exactly 200 unique shapes')
    return shapes
