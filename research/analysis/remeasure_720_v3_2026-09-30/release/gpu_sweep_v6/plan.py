"""Frozen condition generation; no CUDA/PyTorch import needed."""
import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = {
    "version": 4, "seed": 20260919, "blocks": 7,
    "settle_ms": 3000.0, "block_ms": 100.0, "preheat_seconds": 180,
    "max_repeats": 100000, "graph_nodes_max": 64,
    "graph_target_replay_ms": 2.0,
    "graph_batch_policy": "verified64_else_first_probe_then_reuse_per_shape_dtype",
    "verified_graph64_cases": [{"shape": [64, 64, 64], "dtype": "fp16"},
                               {"shape": [64, 64, 64], "dtype": "fp32"},
                               {"shape": [128, 128, 128], "dtype": "fp16"}],
    "pressure_mib": 256, "pressure_samples": 32,
    "relative_iqr_limit": 0.05, "drift_limit": 0.05,
    "between_round_limit": 0.05, "sample_side": 16,
    "tolerances": {"fp32": [2e-5, 2e-4], "fp16": [1e-3, 5e-3]},
    "expected_gpu": "RTX 4000 SFF Ada", "expected_capability": [8, 9],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def task_seed(task):
    # Identical operands across mode/round comparisons, and across pilot/full.
    keys = ("shape", "dtype", "batch", "widths", "mib")
    return int(digest({k: task[k] for k in keys if k in task})[:8], 16)


def graph_key(task):
    return digest({k: task[k] for k in ("shape", "dtype")})


def read_shapes(name):
    provenance = json.loads((ROOT / "inputs" / "provenance.json").read_text())
    observed = hashlib.sha256((ROOT / "inputs" / name).read_bytes()).hexdigest()
    if observed != provenance[name]["sha256"]:
        raise ValueError(f"Input checksum mismatch: {name}")
    with (ROOT / "inputs" / name).open() as f:
        return [tuple(int(r[k]) for k in ("M", "N", "K")) for r in csv.DictReader(f)]


ANCHORS = [(64, 64, 64), (128, 128, 128), (256, 256, 256),
           (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
           (4096, 4096, 4096), (64, 4096, 64), (4096, 64, 64),
           (64, 64, 4096), (65, 129, 257), (1024, 64, 264),
           (1024, 65, 264), (136, 256, 4096), (136, 257, 4096),
           (1313, 545, 249)]


def make_plan(size):
    if size not in ("pilot", "full"):
        raise ValueError(size)
    old = read_shapes("legacy_shapes.csv")
    review = read_shapes("review_shapes.csv")
    pilot_shapes = list(dict.fromkeys(ANCHORS + review[:8]))
    rng = random.Random(PROTOCOL["seed"])
    development = set(pilot_shapes) | set(review)
    eligible = sorted(set(old) - development)
    rng.shuffle(eligible)
    n = len(eligible)
    splits = {s: ("legacy_train" if i < int(n*.7) else
                  "legacy_validation" if i < int(n*.85) else "legacy_test_explored")
              for i, s in enumerate(eligible)}
    splits.update({s: "development" for s in development})
    # Reserve fresh shapes before measurement. No pilot/control selection from these.
    fresh = []
    excluded = set(old) | development
    for axis in range(3):
        for boundary in (128, 256, 512, 1024, 2048):
            for delta in (-8, -1, 0, 1, 8):
                s = [512, 512, 512]
                s[axis] = boundary + delta
                excluded.add(tuple(s))
    for b in (32, 64, 128, 512):
        for w in (128, 256, 512, 1024, 2048):
            for k in (128, 256, 512, 1024, 2048):
                excluded.add((b, w, k))
    excluded.update((w, w, w) for w in (2048, 4096, 8192))
    import math
    while len(fresh) < 1000:
        s = tuple(round(math.exp(rng.uniform(math.log(64), math.log(4096))))
                  for _ in range(3))
        if len(fresh) % 2 == 0:
            s = tuple(max(64, min(4096, round(v/8)*8)) for v in s)
        if s not in excluded:
            fresh.append(s)
            excluded.add(s)
    splits.update({s: "fresh_holdout" for s in fresh})
    main = pilot_shapes if size == "pilot" else list(dict.fromkeys(old + pilot_shapes + fresh))
    controls = pilot_shapes[:4] + pilot_shapes[11:15]
    if size == "full":
        controls = list(dict.fromkeys(ANCHORS + review))
    # One-factor boundary families; keep their neighbours in a single evaluation group.
    families = []
    if size == "full":
        for axis in range(3):
            for boundary in (128, 256, 512, 1024, 2048):
                members = []
                for delta in (-8, -1, 0, 1, 8):
                    s = [512, 512, 512]
                    s[axis] = boundary + delta
                    members.append(tuple(s))
                families.append((f"axis{axis}_boundary{boundary}", members))
    chain_specs = [(64, [256, 512, 128])]
    if size == "full":
        chain_specs = [(b, [w]*(d+1)) for b in (32, 64, 128, 512)
                       for w in (256, 1024, 2048) for d in (2, 4, 8)]
        chain_specs += [(b, [256, 512, 128, 1024, 256]) for b in (64, 128, 512)]
    jobs = []

    def add(**task):
        task["id"] = digest(task)[:24]
        jobs.append(task)

    rounds = 2 if size == "pilot" else 3
    for rd in range(rounds):
        # Shuffle groups each round; reverse dtype and mode order on the next round.
        order = list(main)
        random.Random(PROTOCOL["seed"] + rd).shuffle(order)
        dtypes = ("fp32", "fp16") if rd % 2 == 0 else ("fp16", "fp32")
        paired_modes = ("prealloc", "graph") if rd % 2 == 0 else ("graph", "prealloc")
        for mib in ((256,) if size == "pilot" else (256, 512, 1024)):
            add(kind="bandwidth", mib=mib, round=rd, mode="copy", split="calibration")
        for s in ((2048, 2048, 2048), (4096, 4096, 4096)) if size == "pilot" else (
                (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)):
            for dt in dtypes:
                add(kind="gemm", phase="compute_reference", shape=s, dtype=dt,
                    round=rd, mode="prealloc", split="calibration")
        for s in order:
            for dt in dtypes:
                for mode in paired_modes:
                    add(kind="gemm", phase="main", shape=s, dtype=dt, round=rd,
                        mode=mode, split=splits.get(s, "development"))
        control_order = list(controls)
        random.Random(PROTOCOL["seed"] + 100 + rd).shuffle(control_order)
        modes = ["prealloc", "allocate", "graph", "pressure"]
        if rd % 2:
            modes.reverse()
        for s in control_order:
            for dt in dtypes:
                for mode in modes:
                    add(kind="gemm", phase="controls", shape=s, dtype=dt,
                        round=rd, mode=mode, split="development")
        for group, members in families:
            members = members if rd % 2 == 0 else members[::-1]
            for dt in dtypes:
                for s in members:
                    for mode in paired_modes:
                        add(kind="gemm", phase="boundary", group=group, shape=s,
                            dtype=dt, round=rd, mode=mode, split="boundary_diagnostic")
        for b, widths in chain_specs:
            for dt in dtypes:
                chain_modes = ["prealloc", "sync_each", "graph"]
                if rd % 2:
                    chain_modes.reverse()
                for mode in chain_modes:
                    add(kind="chain", batch=b, widths=widths, dtype=dt,
                        round=rd, mode=mode, split="composition")
    # Profiling is separate from timing. End-of-run subprocesses, not concurrent work.
    profile_shapes = [(1024, 64, 264), (1024, 65, 264)]
    if size == "full":
        profile_shapes += [(136, 256, 4096), (136, 257, 4096),
                           (64, 64, 64), (4096, 4096, 4096)] + review[:4]
    for s in profile_shapes:
        for dt in (("fp16",) if size == "pilot" else ("fp32", "fp16")):
            for cache in ("all", "none"):
                add(kind="profile", tool="ncu", shape=s, dtype=dt,
                    cache=cache, mode="prealloc", split="profile_only")
    for mode in ("prealloc", "sync_each", "graph"):
        add(kind="profile", tool="nsys", batch=64, widths=[1024]*5,
            dtype="fp16", mode=mode, split="profile_only")
    if size == "pilot":
        # The changed path is paired GEMM acquisition. Existing auxiliary experiments remain in full.
        jobs = [t for t in jobs if t.get("phase") == "main"]
    assert len({t["id"] for t in jobs}) == len(jobs)
    return {"size": size, "protocol": PROTOCOL, "tasks": jobs,
            "shape_splits": [{"shape": s, "split": split}
                             for s, split in sorted(splits.items())],
            "fresh_holdout_note": "Do not inspect holdout latency until model selection is frozen.",
            "timing_targets": {"prealloc": "ordinary preallocated GEMM stream elapsed time",
                               "graph": "Graph GEMM stream elapsed time including internal GPU gaps"},
            "round_note": "Separate randomized passes in one process, not independent machine sessions."}
