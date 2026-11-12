"""Standalone FFN and Conv components on actual fixed layer intermediates."""
from layers import tensor_identity


def prepare(torch, case, pair):
    if case['layer'] == 'linear':
        return {}, {'gemm_record': 'decomposed'}
    if case['layer'] == 'ffn':
        x = pair['input']
        p1, p2 = pair['prepared']
        h = torch.mm(x, p1)
        z = torch.nn.functional.gelu(h, approximate='none')
        functions = dict(gemm1=lambda: torch.mm(x, p1),
                         gelu=lambda: torch.nn.functional.gelu(h, approximate='none'),
                         gemm2=lambda: torch.mm(z, p2))
        result = functions['gemm2']()
        torch.testing.assert_close(result, pair['functions']['decomposed'](), rtol=0, atol=0)
        if not all(bool(torch.isfinite(t).all().item()) for t in (h, z, result)):
            raise RuntimeError('Non-finite FFN component output')
        tensors = dict(gemm1_input=x, gemm1_weight=p1, gelu_input=h,
                       gemm2_input=z, gemm2_weight=p2)
        return functions, {name: tensor_identity(tensor) for name, tensor in tensors.items()}
    x, packed = pair['input'], pair['prepared'][0]
    b, h, w, ci, co = (case[k] for k in ('batch', 'height', 'width', 'cin', 'cout'))
    if case['kernel'] == 1:
        def input_pack():
            return x.permute(0, 2, 3, 1).contiguous().view(b * h * w, ci)
        a = input_pack()
        functions = {'input_pack': input_pack}
        intermediate = {'input_pack_input': x, 'mm_input': a, 'mm_weight': packed}
    else:
        def unfold():
            return torch.nn.functional.unfold(x, 3, padding=1)
        patches = unfold()
        def input_pack():
            return patches.transpose(1, 2).contiguous().view(b * h * w, ci * 9)
        a = input_pack()
        functions = {'unfold': unfold, 'input_pack': input_pack}
        intermediate = {'unfold_input': x, 'input_pack_input': patches,
                        'mm_input': a, 'mm_weight': packed}
    y = torch.mm(a, packed)
    def output_pack():
        return y.view(b, h, w, co).permute(0, 3, 1, 2).contiguous()
    functions.update(mm=lambda: torch.mm(a, packed), output_pack=output_pack)
    intermediate['output_pack_input'] = y
    expected = pair['functions']['decomposed']()
    torch.testing.assert_close(output_pack(), expected, rtol=0, atol=0)
    return functions, {name: tensor_identity(value) for name, value in intermediate.items()}
