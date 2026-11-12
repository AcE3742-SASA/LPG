"""One runnable check: frozen coverage, correctness math and quality gates."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import compare
import select_cases
from layers import make_pair, sampled_reference, correctness, seed_for


def check_analysis(config,cases):
    def write(path,value):
        path.write_text(json.dumps(value))
    sensor={'clocks.sm':720,'clocks.mem':6801,'clocks_event_reasons.sw_power_cap':False}
    telemetry=SimpleNamespace(read=lambda *args:([sensor],{}),
                              window=lambda *args:dict(inside=[sensor]))
    loaded=dict(suite=SimpleNamespace(atomic_json=write),telemetry=telemetry)
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        (out/'worker').mkdir(); (out/'records').mkdir()
        write(out/'worker/telemetry_metadata.json',{'timezone_offset_seconds':0})
        write(out/'correct.json',{'passed':True})
        selection=[c for c in cases if c['id'] in config['health_case_ids']]
        def add_records(selection):
            for case in selection:
                for dtype in config['dtypes']:
                    for rd in range(3):
                        for implementation in ('standard','decomposed'):
                            row_id=compare.key(case,dtype,rd,implementation)
                            write(out/'records'/f'{row_id}.json',dict(id=row_id,case=case,dtype=dtype,
                                timing=dict(median_ms=1 if implementation=='standard' else 2,flags=[],
                                    blocks=[dict(host_envelope_start_utc='2026-10-06T00:00:00+00:00',
                                                 host_envelope_end_utc='2026-10-06T00:00:01+00:00')]),
                                memory=dict(extra_peak_allocated_bytes=100),prepared_weight_bytes=20,
                                correctness_before='correct.json',correctness_after='correct.json'))
        add_records(selection)
        healthy=compare.analyze(out,config,cases,loaded,health_only=True)
        assert healthy['gate']=='PASS' and healthy['recorded']==48
        assert all(c['ratio']==2 and c['difference_ms']==1 for c in healthy['comparisons'])
        sensor['clocks.sm']=705
        assert compare.analyze(out,config,cases,loaded,health_only=True)['gate']=='REVIEW'
        sensor['clocks.sm']=720
        write(out/'correct.json',{'passed':False})
        assert compare.analyze(out,config,cases,loaded,health_only=True)['gate']=='FAIL'
        write(out/'correct.json',{'passed':True})
        write(out/'health_check.json',{'gate':'REVIEW'})
        write(out/'status.json',{'restoration':'COMMANDS_ACCEPTED','collection':'COMPLETE'})
        partial=compare.analyze(out,config,cases,loaded)
        assert partial['gate']=='REVIEW' and partial['collection']=='PRECHECK_STOPPED'
        assert partial['missing']==1920-48
        write(out/'status.json',{'restoration':'FAILED','collection':'COMPLETE'})
        assert compare.analyze(out,config,cases,loaded)['gate']=='FAIL'
        write(out/'status.json',{'restoration':'COMMANDS_ACCEPTED','collection':'FAILED_OR_INTERRUPTED'})
        assert compare.analyze(out,config,cases,loaded)['gate']=='FAIL'
        (out/'correct.json').rename(out/'missing_correct.json')
        assert compare.analyze(out,config,cases,loaded,health_only=True)['gate']=='FAIL'
        (out/'missing_correct.json').rename(out/'correct.json')
        add_records([c for c in cases if c['id'] not in config['health_case_ids']])
        write(out/'health_check.json',{'gate':'PASS'})
        write(out/'status.json',{'restoration':'COMMANDS_ACCEPTED','collection':'COMPLETE'})
        assert compare.analyze(out,config,cases,loaded)['gate']=='FAIL'
        write(out/'worker/complete.json',{'complete':True})
        complete=compare.analyze(out,config,cases,loaded)
        assert complete['gate']=='PASS' and complete['recorded']==1920
        assert len(complete['comparisons'])==320
        assert len((out/'comparison.csv').read_text().splitlines())==321
        write(out/'failure.json',{'type':'RuntimeError','message':'Injected failure'})
        assert compare.analyze(out,config,cases,loaded)['gate']=='FAIL'


def check_cpu(config,cases):
    import torch
    torch.set_num_threads(2)
    probes=[dict(layer='conv1',batch=2,height=3,width=5,cin=3,cout=5,kernel=1),
            dict(layer='conv3',batch=2,height=3,width=5,cin=3,cout=5,kernel=3),
            dict(layer='conv3',batch=1,height=5,width=7,cin=63,cout=65,kernel=3),
            dict(layer='linear',rows=17,input_width=19,output_width=23),
            dict(layer='ffn',rows=17,input_width=19,hidden_width=31)]
    probes += [c for c in cases if c['id'] in config['health_case_ids']]
    with torch.inference_mode():
        for case in probes:
            for dtype in config['dtypes']:
                pair=make_pair(torch,case,dtype,device='cpu')
                result=correctness(torch,case,pair,dtype,config['tolerances'])
                assert result['passed'], (case,dtype,result)
                rows,cols,reference=sampled_reference(torch,case,pair)
                x=pair['input'].double()
                weights=[w.double() for w in pair['weights']]
                if case['layer'].startswith('conv'):
                    whole=torch.nn.functional.conv2d(x,weights[0],padding=case['kernel']//2)
                    flat=whole.permute(0,2,3,1).reshape(-1,case['cout'])
                elif case['layer']=='linear':
                    flat=torch.nn.functional.linear(x,weights[0])
                else:
                    flat=torch.nn.functional.linear(torch.nn.functional.gelu(
                        torch.nn.functional.linear(x,weights[0]),approximate='none'),weights[1])
                torch.testing.assert_close(reference,flat[rows][:,cols],rtol=1e-12,atol=1e-12)
                pair['prepared'][0].zero_()
                assert not correctness(torch,case,pair,dtype,config['tolerances'])['passed']
                assert seed_for(case,dtype)==make_pair(torch,case,dtype,device='cpu')['seed']
    print(f'CPU PASS: {len(probes)*2} dtype/path checks, independent full FP64 references '
          'and injected weight failures; torch='+torch.__version__)


if __name__=='__main__':
    config,cases=compare.verify()
    assert len(cases)==160 and len({c['id'] for c in cases})==160
    assert compare.forbidden_overlap((64,64,63),{(64,64,64)})
    assert not compare.forbidden_overlap((64,64,55),{(64,64,64)})
    reserved={tuple(v) for v in compare.read(compare.HERE/'forbidden_shapes.json')}
    assert select_cases.select(reserved,config['seed'])[0]==cases
    compare.load_frozen()
    check_analysis(config,cases)
    print('STATIC PASS: coverage, frozen dependencies, reserved shapes, ratios and failure gates')
    if '--cpu' in sys.argv:
        check_cpu(config,cases)
