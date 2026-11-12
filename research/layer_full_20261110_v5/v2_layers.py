"""Standard nn modules and the declared explicit GEMM paths. No custom kernels."""
import hashlib
import json
import random


def seed_for(case, dtype):
    value = json.dumps([case, dtype], sort_keys=True, separators=(',', ':'))
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def make_pair(torch, case, dtype, device='cuda'):
    dt = {'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    gen = torch.Generator(device=device).manual_seed(seed_for(case, dtype))
    def rand(shape):
        return torch.randn(shape, dtype=dt, device=device, generator=gen) * .1
    kind = case['layer']
    if kind.startswith('conv'):
        b, h, w, ci, co = (case[k] for k in ('batch', 'height', 'width', 'cin', 'cout'))
        kernel = case['kernel']
        x, weight = rand((b, ci, h, w)), rand((co, ci, kernel, kernel))
        module = torch.nn.Conv2d(ci, co, kernel, padding=kernel//2, bias=False,
                                dtype=dt, device='meta')
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        packed = weight.reshape(co, -1).t().contiguous()
        def decomposed():
            if kernel == 1:
                a = x.permute(0, 2, 3, 1).contiguous().view(b*h*w, ci)
            else:
                patches = torch.nn.functional.unfold(x, kernel, padding=kernel//2)
                a = patches.transpose(1, 2).contiguous().view(b*h*w, ci*kernel*kernel)
            y = torch.mm(a, packed)
            return y.view(b, h, w, co).permute(0, 3, 1, 2).contiguous()
        weights, prepared = [weight], [packed]
    else:
        m, d = case['rows'], case['input_width']
        x = rand((m, d))
        if kind == 'linear':
            n = case['output_width']
            weight = rand((n, d))
            module = torch.nn.Linear(d, n, bias=False, dtype=dt, device='meta')
            module.weight = torch.nn.Parameter(weight, requires_grad=False)
            packed = weight.t().contiguous()
            def decomposed():
                return torch.mm(x, packed)
            weights, prepared = [weight], [packed]
        elif kind == 'ffn':
            f = case['hidden_width']
            first, second = rand((f, d)), rand((d, f))
            module = torch.nn.Sequential(
                torch.nn.Linear(d, f, bias=False, dtype=dt, device='meta'),
                torch.nn.GELU(approximate='none'),
                torch.nn.Linear(f, d, bias=False, dtype=dt, device='meta'))
            module[0].weight = torch.nn.Parameter(first, requires_grad=False)
            module[2].weight = torch.nn.Parameter(second, requires_grad=False)
            p1, p2 = first.t().contiguous(), second.t().contiguous()
            def decomposed():
                return torch.mm(torch.nn.functional.gelu(torch.mm(x, p1), approximate='none'), p2)
            weights, prepared = [first, second], [p1, p2]
        else:
            raise ValueError('Unknown layer: ' + kind)
    module.eval()
    return dict(functions={'standard': lambda: module(x), 'decomposed': decomposed},
                input=x, weights=weights, prepared=prepared, seed=seed_for(case, dtype),
                prepared_weight_bytes=sum(t.numel()*t.element_size() for t in prepared))


def indices(size, seed, side=16):
    if size <= side:
        return list(range(size))
    anchors = {0, size-1, size//2}
    rng = random.Random(seed)
    while len(anchors) < side:
        anchors.add(rng.randrange(size))
    return sorted(anchors)


def sampled_reference(torch, case, pair, round_intermediates=True):
    x = pair['input'].detach().to(device='cpu', dtype=torch.float64)
    weights = [t.detach().to(device='cpu', dtype=torch.float64) for t in pair['weights']]
    if case['layer'].startswith('conv'):
        b, ci, h, w = x.shape
        kernel = case['kernel']
        rows = indices(b*h*w, pair['seed'])
        cols = indices(case['cout'], pair['seed'] ^ 12345)
        padded = torch.nn.functional.pad(x, (kernel//2,)*4)
        patches = []
        for row in rows:
            batch, pos = divmod(row, h*w)
            yy, xx = divmod(pos, w)
            patches.append(padded[batch, :, yy:yy+kernel, xx:xx+kernel].reshape(-1))
        reference = torch.stack(patches) @ weights[0][cols].reshape(len(cols), -1).t()
    else:
        rows = indices(case['rows'], pair['seed'])
        cols = indices(case.get('output_width', case['input_width']), pair['seed'] ^ 12345)
        if case['layer'] == 'linear':
            reference = x[rows] @ weights[0][cols].t()
        else:
            z = x[rows] @ weights[0].t()
            storage_rounding = round_intermediates and pair['input'].dtype == torch.float16
            # Each unfused FFN operator stores its output in the input dtype.
            if storage_rounding:
                z = z.to(torch.float16).to(torch.float64)
            z = torch.nn.functional.gelu(z, approximate='none')
            if storage_rounding:
                z = z.to(torch.float16).to(torch.float64)
            reference = z @ weights[1][cols].t()
    return rows, cols, reference


def correctness(torch, case, pair, dtype, tolerances):
    outputs = {name: fn() for name, fn in pair['functions'].items()}
    expected_shape = ((case['batch'], case['cout'], case['height'], case['width'])
                      if case['layer'].startswith('conv') else
                      (case['rows'], case.get('output_width', case['input_width'])))
    if any(tuple(t.shape) != expected_shape or not t.is_contiguous() for t in outputs.values()):
        raise RuntimeError('Output shape or contiguous layout differs from the contract')
    rows, cols, reference = sampled_reference(torch, case, pair)
    rounded_ffn = case['layer'] == 'ffn' and dtype == 'fp16'
    mathematical = sampled_reference(torch, case, pair, False)[2] if rounded_ffn else reference
    atol, rtol = tolerances[dtype]
    results = {}
    for name, output in outputs.items():
        finite = bool(torch.isfinite(output).all().item())
        if case['layer'].startswith('conv'):
            flat = output.permute(0, 2, 3, 1).reshape(-1, case['cout'])
        else:
            flat = output
        got = flat[rows][:, cols].to(device='cpu', dtype=torch.float64)
        error = (got-reference).abs()
        bound = atol + rtol*reference.abs()
        results[name] = dict(passed=finite and bool((error <= bound).all().item()),
                             finite_all=finite, points=len(rows)*len(cols),
                             max_abs_error=error.max().item(),
                             max_tolerance_ratio=(error/bound).max().item(),
                             output_shape=list(output.shape), output_stride=list(output.stride()))
        if not results[name]['passed']:
            results[name].update(reference=reference.tolist(), observed=got.tolist())
        if rounded_ffn:
            error_math = (got-mathematical).abs()
            bound_math = atol + rtol*mathematical.abs()
            results[name]['mathematical_fp64_diagnostic'] = dict(
                passed=finite and bool((error_math <= bound_math).all().item()),
                max_abs_error=error_math.max().item(),
                max_tolerance_ratio=(error_math/bound_math).max().item())
    a, b = (outputs[k].float() for k in ('standard', 'decomposed'))
    difference = (a-b).abs()
    pair_pass = bool((difference <= atol + rtol*a.abs()).all().item())
    return dict(passed=pair_pass and all(r['passed'] for r in results.values()),
                full_pair_passed=pair_pass, pair_max_abs_error=difference.max().item(),
                reference=('CPU FP64 with FP16 storage rounding after GEMM1 and GELU'
                           if rounded_ffn else 'CPU FP64 mathematical forward from actual rounded input/weights'),
                reference_scope='fixed sampled output points; full-output finite and pair checks',
                rows=rows, columns=cols, atol=atol, rtol=rtol, implementations=results)


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
