"""CUDA measurements. Correctness/profiling work never enters a timing block."""
import math
import datetime as dt
import random
import statistics
import time

from plan import task_seed


def quantile(values, q):
    a = sorted(values)
    i = (len(a)-1)*q
    lo = int(i)
    return a[lo] + (a[min(lo+1, len(a)-1)]-a[lo])*(i-lo)


def summarize(blocks, protocol):
    values = [b["latency_ms"] for b in blocks]
    med = statistics.median(values)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Non-positive/non-finite CUDA elapsed time")
    iqr = (quantile(values, .75)-quantile(values, .25))/med
    drift = (values[-1]-values[0])/values[0]
    flags = []
    if iqr > protocol["relative_iqr_limit"]:
        flags.append("relative_iqr")
    if abs(drift) > protocol["drift_limit"]:
        flags.append("first_last_drift")
    return {"median_ms": med, "relative_iqr": iqr, "first_last_drift": drift,
            "min_ms": min(values), "max_ms": max(values), "flags": flags}


def configure(torch):
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        matmul.fp32_precision = "ieee"
        values = {"fp32_precision": matmul.fp32_precision}
    else:
        matmul.allow_tf32 = False
        values = {"allow_tf32": matmul.allow_tf32}
    for name in ("allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation"):
        if hasattr(matmul, name):
            setattr(matmul, name, False)
            values[name] = getattr(matmul, name)
    return values


class Timer:
    def __init__(self, torch, protocol):
        self.torch, self.p = torch, protocol

    def block(self, fn, repeats, operations=1):
        t = self.torch
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        # Initialize event objects outside the observed interval.
        start.record(); end.record(); end.synchronize()
        wall = time.perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter()-wall)*1000
        elapsed = start.elapsed_time(end)
        return {"timestamp": timestamp, "repeats": repeats, "operations_per_call": operations,
                "elapsed_ms": elapsed, "host_wall_ms": wall_ms,
                "latency_ms": elapsed/(repeats*operations),
                "host_latency_ms": wall_ms/(repeats*operations)}

    def prepare(self, fn, operations=1):
        probe = self.block(fn, 3, operations)
        if probe["elapsed_ms"] <= 0:
            raise RuntimeError("CUDA probe elapsed time is zero")
        repeats = max(1, min(self.p["max_repeats"], math.ceil(
            self.p["block_ms"]/(probe["elapsed_ms"]/3))))
        started = time.perf_counter()
        settle = []
        elapsed = 0
        # Accumulate OBSERVED event time, not probe-estimated warmup iterations.
        while elapsed < self.p["settle_ms"]:
            b = self.block(fn, repeats, operations)
            settle.append(b)
            elapsed += b["elapsed_ms"]
        calibration = []
        for _ in range(4):
            b = self.block(fn, repeats, operations)
            calibration.append(b)
            # Leave margin above the 80% short-block quality threshold.
            if .9*self.p["block_ms"] <= b["elapsed_ms"] <= 1.1*self.p["block_ms"]:
                break
            new = max(1, min(self.p["max_repeats"], round(
                repeats*self.p["block_ms"]/max(b["elapsed_ms"], 1e-9))))
            if repeats == new:
                break
            repeats = new
        return repeats, {"settle_event_ms": elapsed,
                         "prepare_wall_ms": (time.perf_counter()-started)*1000,
                         "settle_blocks": settle, "calibration": calibration}

    def run(self, fn, operations=1):
        repeats, setup = self.prepare(fn, operations)
        blocks = [self.block(fn, repeats, operations) for _ in range(self.p["blocks"])]
        stats = summarize(blocks, self.p)
        if any(b["elapsed_ms"] < .8*self.p["block_ms"] for b in blocks):
            stats["flags"].append("short_block")
        return {"setup": setup, "blocks": blocks, **stats}


def capture(torch, fn, timer):
    probe = timer.block(fn, 3)
    nodes = max(1, min(timer.p["graph_nodes_max"], math.ceil(2/probe["latency_ms"])))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, nodes


def check_mm(torch, a, b, c, dtype, seed, protocol):
    m, n = c.shape
    rng = random.Random(seed)
    side = protocol["sample_side"]
    rows = list(range(m)) if m*n <= 4096 else sorted(rng.sample(range(m), min(m, side)))
    cols = list(range(n)) if m*n <= 4096 else sorted(rng.sample(range(n), min(n, side)))
    # CPU FP64 reference of the actual rounded input tensors, not regenerated FP64 inputs.
    aa = a[rows, :].to(device="cpu", dtype=torch.float64)
    bb = b[:, cols].to(device="cpu", dtype=torch.float64)
    got = c[rows, :][:, cols].to(device="cpu", dtype=torch.float64)
    ref = aa @ bb
    diff = (got-ref).abs()
    atol, rtol = protocol["tolerances"][dtype]
    bound = atol + rtol*ref.abs()
    finite = bool(torch.isfinite(c).all().item())
    passed = finite and bool((diff <= bound).all().item())
    result = {"passed": passed, "finite_all": finite, "scope": "full" if m*n <= 4096 else "sampled",
              "points": len(rows)*len(cols), "rows": rows, "cols": cols,
              "atol": atol, "rtol": rtol,
              "max_abs_error": diff.max().item(),
              "max_tolerance_ratio": (diff/bound).max().item(),
              "relative_l2_error": (torch.linalg.vector_norm(diff)/
                                    torch.linalg.vector_norm(ref).clamp_min(1e-30)).item()}
    if not passed:
        result.update(reference=ref.tolist(), observed=got.tolist())
    return result


def make_gemm(torch, task):
    m, n, k = task["shape"]
    dt = {"fp16": torch.float16, "fp32": torch.float32}[task["dtype"]]
    gen = torch.Generator(device="cuda").manual_seed(task_seed(task))
    a = torch.randn((m, k), device="cuda", dtype=dt, generator=gen)*.1
    b = torch.randn((k, n), device="cuda", dtype=dt, generator=gen)*.1
    out = [torch.empty((m, n), device="cuda", dtype=dt)]

    def fn():
        if task["mode"] == "allocate":
            out[0] = torch.mm(a, b)
        else:
            torch.mm(a, b, out=out[0])
    fn()
    return a, b, out, fn


def measure_gemm(torch, task, protocol):
    a, b, out, fn = make_gemm(torch, task)
    timer = Timer(torch, protocol)
    graph = None
    if task["mode"] == "graph":
        graph, nodes = capture(torch, fn, timer)
        timing = timer.run(graph.replay, nodes)
        timing["graph_nodes"] = nodes
    elif task["mode"] == "pressure":
        # Pressure is an experimental intervention, not proof of a cold cache.
        generator = torch.Generator(device="cuda").manual_seed(task_seed(task)+1)
        pressure = torch.randint(0, 256, (protocol["pressure_mib"]*1024**2,),
                                 dtype=torch.uint8, device="cuda", generator=generator)
        def pressured():
            pressure.add_(1)
            fn()
        _, setup = timer.prepare(pressured)
        setup["meaning"] = "Observed stabilization of pressure plus GEMM, not GEMM-only latency"
        blocks = []
        for _ in range(protocol["blocks"]):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(protocol["pressure_samples"])]
            ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
            for s, e in zip(starts, ends):
                s.record(); e.record()
            torch.cuda.synchronize()
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            wall = time.perf_counter()
            for s, e in zip(starts, ends):
                pressure.add_(1)
                s.record()
                fn()
                e.record()
            torch.cuda.synchronize()
            samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            blocks.append({"timestamp": timestamp, "latency_ms": statistics.mean(samples),
                           "elapsed_ms": sum(samples), "repeats": len(samples),
                           "operations_per_call": 1, "samples_ms": samples,
                           "host_wall_with_pressure_ms": (time.perf_counter()-wall)*1000})
        timing = {"setup": setup, "blocks": blocks, **summarize(blocks, protocol),
                  "pressure_mib": protocol["pressure_mib"]}
    else:
        timing = timer.run(fn)
    check = check_mm(torch, a, b, out[0], task["dtype"], task_seed(task), protocol)
    m, n, k = task["shape"]
    flops = 2*m*n*k
    ideal_bytes = (m*k+k*n+m*n)*a.element_size()
    return {"timing": timing, "correctness": check, "flops": flops,
            "ideal_bytes": ideal_bytes, "ideal_oi": flops/ideal_bytes,
            "tflops": flops/(timing["median_ms"]*1e9),
            "input_seed": task_seed(task)}


def make_chain(torch, task):
    dt = {"fp16": torch.float16, "fp32": torch.float32}[task["dtype"]]
    gen = torch.Generator(device="cuda").manual_seed(task_seed(task))
    widths, batch = task["widths"], task["batch"]
    xs = [torch.randn((batch, widths[0]), device="cuda", dtype=dt, generator=gen)*.1]
    weights = []
    for k, n in zip(widths, widths[1:]):
        weights.append(torch.randn((k, n), device="cuda", dtype=dt, generator=gen)/math.sqrt(k))
        xs.append(torch.empty((batch, n), device="cuda", dtype=dt))
    parts = [lambda i=i: torch.mm(xs[i], weights[i], out=xs[i+1]) for i in range(len(weights))]

    def whole():
        for i, part in enumerate(parts):
            part()
            if task["mode"] == "sync_each" and i < len(parts)-1:
                torch.cuda.synchronize()
    whole()
    return xs, weights, parts, whole


def measure_chain(torch, task, protocol):
    xs, weights, parts, whole = make_chain(torch, task)
    timer = Timer(torch, protocol)
    calls = parts + [whole]
    graphs = []
    nodes = [1]*len(calls)
    if task["mode"] == "graph":
        for i, fn in enumerate(calls):
            g, nodes[i] = capture(torch, fn, timer)
            graphs.append(g)
        calls = [g.replay for g in graphs]
    paired = []
    for block in range(protocol["blocks"]):
        order = list(range(len(calls)))
        random.Random(task_seed(task)+task["round"]*100+block).shuffle(order)
        values = {}
        for i in order:
            repeats, setup = timer.prepare(calls[i], nodes[i])
            values[str(i)] = {**timer.block(calls[i], repeats, nodes[i]), "setup": setup}
        iso = sum(values[str(i)]["latency_ms"] for i in range(len(parts)))
        chain = values[str(len(parts))]["latency_ms"]
        paired.append({"order": order, "measurements": values,
                       "isolated_sum_ms": iso, "chain_ms": chain,
                       "composition_residual_ms": chain-iso,
                       "composition_relative": (chain-iso)/chain})
    # Verify each layer using its actual input; this is not a whole-chain FP64 error bound.
    whole()
    checks = [check_mm(torch, xs[i], weights[i], xs[i+1], task["dtype"],
                       task_seed(task)+i, protocol) for i in range(len(parts))]
    timings = [values["measurements"][str(len(parts))] for values in paired]
    stats = summarize(timings, protocol)
    for i in range(len(parts)):
        part_stats = summarize([p["measurements"][str(i)] for p in paired], protocol)
        stats["flags"].extend(f"layer{i}_{s}" for s in part_stats["flags"])
    return {"timing": {"blocks": timings, **stats}, "paired_blocks": paired,
            "graph_nodes": nodes, "input_seed": task_seed(task),
            "correctness": {"passed": all(c["passed"] for c in checks),
                            "scope": "sampled_each_layer", "layers": checks}}


def measure_bandwidth(torch, task, protocol):
    count = task["mib"]*1024**2//4
    gen = torch.Generator(device="cuda").manual_seed(task_seed(task))
    a = torch.randn(count, device="cuda", generator=gen)
    b = torch.empty_like(a)
    timing = Timer(torch, protocol).run(lambda: b.copy_(a))
    passed = bool(torch.equal(a, b))
    payload = 2*a.numel()*a.element_size()
    return {"timing": timing, "correctness": {"passed": passed, "scope": "full_copy"},
            "payload_bytes_read_plus_write": payload,
            "effective_copy_gbps": payload/(timing["median_ms"]*1e6),
            "meaning": "Device copy payload bandwidth reference, not measured DRAM traffic."}


def measure(torch, task, protocol):
    with torch.inference_mode():
        if task["kind"] == "gemm":
            return measure_gemm(torch, task, protocol)
        if task["kind"] == "chain":
            return measure_chain(torch, task, protocol)
        if task["kind"] == "bandwidth":
            return measure_bandwidth(torch, task, protocol)
    raise ValueError(task["kind"])
