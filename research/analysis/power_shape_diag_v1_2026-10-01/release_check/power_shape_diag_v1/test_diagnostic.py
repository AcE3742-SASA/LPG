#!/usr/bin/env python3
"""CPU-only checks of the diagnostic's measurement, telemetry and cleanup paths."""
import contextlib
import datetime as dt
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import diagnose as d


def raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError('Expected '+kind.__name__)


def main():
    loaded, config = d.load()
    suite, measure, telemetry, plan = [loaded[k] for k in ('suite','measure','telemetry','plan')]
    assert len(d.batches(config))*config['continuous_trials'] == 128
    assert [case for _,pos,case in d.batches(config) if pos==0] == [0,1,1,0]
    assert all(plan.task_seed(t)==plan.task_seed(dict(t,id='diagnostic',round=99)) for t in config['tasks'])
    for p in d.HERE.glob('*'):
        if p.suffix in ('.py','.sh'):
            p.read_bytes().decode('ascii')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpu = loaded['baseline']['environment_signature']['gpu_uuid']
        suite.atomic_json(root/'telemetry_metadata.json',dict(timezone_offset_seconds=0))
        now = dt.datetime.now(dt.timezone.utc)
        values = [now.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3],gpu,'P0','53','99','705','6801',
                  '69.9','70','0x4','Active','Not Active','Not Active','0','0','0']
        (root/'telemetry.csv').write_text(','.join(values)+'\n'+','.join(values[:6]))
        guard = d.Guard(root,telemetry,gpu,NS(poll=lambda:None))
        row = guard.check()
        assert row['clocks.sm']==705 and row['clocks_event_reasons.sw_power_cap']
        with (root/'telemetry.csv').open('a') as f:
            f.write(','+','.join(values[6:])+'\n')
        assert guard.check()==row
        assert len(guard.new_rows)==1
        for edits in ({'power.limit':65},{'temperature.gpu':85},
                      {'clocks_event_reasons.active':8},{'uuid':'other'},
                      {'clocks_event_reasons.sw_thermal_slowdown':True},{'utilization.gpu':None}):
            raises(RuntimeError,lambda:d.safe_row(dict(row,**edits),gpu))
        guard.process = NS(poll=lambda:1)
        raises(RuntimeError,guard.check)
        guard.process = None
        with patch.object(d.time,'time',return_value=now.timestamp()+10):
            raises(RuntimeError,guard.check)
        # A high-temperature sample between polls must reset the recovery interval.
        clock, samples = [0.0], []
        class RecoveryGuard:
            def check(self):
                clock[0] += 1
                stamp = int(clock[0])
                self.new_rows = [dict(utc=dt.datetime.fromtimestamp(stamp,dt.timezone.utc).isoformat(),
                                      **{'temperature.gpu':52})]
                if stamp==6:
                    self.new_rows.insert(0,dict(utc=dt.datetime.fromtimestamp(5.9,dt.timezone.utc).isoformat(),
                                               **{'temperature.gpu':56}))
                return self.new_rows[-1]
        with patch.object(d.time,'monotonic',side_effect=lambda:clock[0]), patch.object(d.time,'sleep'), patch.object(d.time,'time',return_value=0):
            recovered = d.recover(RecoveryGuard(),dict(config,cooldown_timeout_s=30))
        assert recovered['below_threshold_seconds']==10 and clock[0]==16
        clock[0]=0
        with patch.object(d.time,'monotonic',side_effect=lambda:clock[0]), patch.object(d.time,'sleep'), patch.object(d.time,'time',return_value=0):
            raises(TimeoutError,lambda:d.recover(RecoveryGuard(),dict(config,cooldown_timeout_s=5)))
        # Drive the actual worker and original prepare/run protocol with CPU event substitutes.
        case = root/'worker'
        case.mkdir()
        manifest = dict(source_hashes=d.hashes(),protocol=config,telemetry_fields=telemetry.FIELDS,
                        environment_signature=loaded['baseline']['environment_signature'])
        suite.atomic_json(root/'manifest.json',manifest)
        env = manifest['environment_signature']
        allocations, mm_calls, checked_seeds, capture_events = [], [], [], []
        fake_torch = NS(cuda=NS(mem_get_info=lambda:(8*1024**3,8*1024**3),synchronize=lambda:None,
            nvtx=NS(range=lambda label:contextlib.nullcontext()),
            cudart=lambda:NS(cudaProfilerStart=lambda:capture_events.append('start'),
                            cudaProfilerStop=lambda:capture_events.append('stop'))),
            inference_mode=contextlib.nullcontext)
        class FakeTimer(measure.Timer):
            def block(self,fn,repeats,operations=1):
                fn()
                return dict(timestamp=suite.now(),repeats=repeats,operations_per_call=operations,
                            elapsed_ms=repeats*.1,host_wall_ms=repeats*.1,latency_ms=.1,host_latency_ms=.1)
        def make_gemm(torch,task):
            allocations.append(task)
            return 'a','b',['c'],lambda:mm_calls.append('mm')
        def correctness(torch,a,b,c,dtype,seed,protocol):
            assert (a,b,c,dtype)==('a','b','c','fp32')
            checked_seeds.append(seed)
            return dict(passed=True)
        args = NS(output=root,directory=case,case=0,tool=None)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(d.sys.modules,{'torch':fake_torch}))
            stack.enter_context(patch.object(suite,'environment',return_value=env))
            stack.enter_context(patch.object(loaded['audit'],'check_processes'))
            stack.enter_context(patch.object(measure,'Timer',FakeTimer))
            stack.enter_context(patch.object(measure,'make_gemm',side_effect=make_gemm))
            stack.enter_context(patch.object(measure,'check_mm',side_effect=correctness))
            stack.enter_context(patch.object(d,'Guard',return_value=NS(check=lambda:None)))
            stack.enter_context(patch.object(d,'recover',return_value={'start_sample':row}))
            d.worker(args,loaded,config)
            records = [d.read(p) for p in sorted(case.glob('trial_*.json'))]
            assert len(records)==16 and len(allocations)==1 and len(checked_seeds)==17
            assert all(r['timing']['setup']['settle_event_ms']>=3000 and len(r['timing']['blocks'])==7 for r in records)
            assert all(s==plan.task_seed(config['tasks'][0]) for s in checked_seeds)
            assert len(d.read(case/'all_event_blocks.json'))==624
            for tool in ('ncu','nsys'):
                args.directory=root/tool
                args.directory.mkdir()
                args.tool=tool
                before=len(mm_calls)
                d.worker(args,loaded,config)
                # Prepare: one probe, 30 settling blocks, one calibration; capture: 1 or 20 calls.
                assert len(mm_calls)-before==32+(1 if tool=='ncu' else 20)
                assert d.read(args.directory/'worker_complete.json')['complete']
            assert capture_events==['start','stop','start','stop']
        # Exercise actual control flow and real two-reset helper under failures/interrupts.
        for fault in ('admission','memory','graphics','worker','interrupt','reset',None):
            events=[]
            @contextlib.contextmanager
            def lock(path):
                events.append('lock')
                try:
                    yield
                finally:
                    events.append('unlock')
            @contextlib.contextmanager
            def credentials(*args):
                yield NS(check=lambda:None)
            def admission(*args):
                if fault=='admission':
                    raise RuntimeError('admission')
            def clock_command(suite,out,smi,gpu,name,*options):
                events.append(name)
                code=1 if (name=='set_'+str(fault) or fault=='reset' and name=='reset_graphics') else 0
                return dict(exit_code=code)
            def operation(cred):
                events.append('worker')
                if fault=='worker':
                    raise RuntimeError('worker')
                if fault=='interrupt':
                    raise KeyboardInterrupt('interrupt')
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(d.shutil,'which',return_value='/usr/bin/nvidia-smi'))
                stack.enter_context(patch.object(suite,'lock',side_effect=lock))
                stack.enter_context(patch.object(suite,'command_text',return_value={'exit_code':0}))
                stack.enter_context(patch.object(loaded['audit'],'check_processes',side_effect=admission))
                stack.enter_context(patch.object(loaded['steps'],'require_power'))
                stack.enter_context(patch.object(loaded['steps'],'snapshot'))
                stack.enter_context(patch.object(loaded['steps'],'Credentials',side_effect=credentials))
                stack.enter_context(patch.object(loaded['hold'],'clock_command',side_effect=clock_command))
                call=lambda:d.controlled_run(root,loaded,config,operation)
                if fault:
                    raises(KeyboardInterrupt if fault=='interrupt' else RuntimeError,call)
                else:
                    call()
            assert events[0]=='lock' and events[-1]=='unlock'
            if fault=='admission':
                assert not any(e.startswith(('set_','reset_')) for e in events)
            else:
                assert events[-3:]==['reset_graphics','reset_memory','unlock']
            if fault in ('admission','memory','graphics'):
                assert 'worker' not in events
            assert d.read(root/'status.json')['collection']==('COMPLETE' if fault is None else 'FAILED_OR_INTERRUPTED')
        with patch.object(d.shutil,'which',side_effect=lambda tool:'/usr/bin/'+tool):
            cmd=d.profile_command('ncu',['python','worker'],root,config)
            assert cmd[cmd.index('--clock-control')+1]=='none'
            assert cmd[cmd.index('--cache-control')+1]=='none'
            assert cmd.count('--section')==5
            cmd=d.profile_command('nsys',['python','worker'],root,config)
            assert '--sample=none' in cmd and '--cpuctxsw=none' in cmd
        from profiling import ncu_has_kernel_data, nsys_has_kernel_data
        assert not ncu_has_kernel_data('ID,Kernel Name,gpu__time_duration.sum\n')
        assert ncu_has_kernel_data('ID,Kernel Name,gpu__time_duration.sum\n0,sgemm,100\n')
        assert ncu_has_kernel_data('ID,Kernel Name,Metric Name,Metric Value\n0,sgemm,sm__throughput,12\n')
        assert not nsys_has_kernel_data('Instances,Total Time (ns),Name\n0,0,sgemm\n')
        assert nsys_has_kernel_data('Instances,Total Time (ns),Name\n20,100,sgemm\n')
        stats=d.window_stats([row],row['utc'],row['utc'])
        assert stats['sw_power_cap_samples']==1 and stats['below_target_samples']==1
        assert not stats['memory_clock_mismatch']
    # Recompute a complete synthetic return, then prove missing profiles and poor matching stay visible.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        suite.atomic_json(root/'manifest.json',dict(telemetry_fields=telemetry.FIELDS))
        suite.atomic_json(root/'status.json',dict(collection='COMPLETE',restoration='COMMANDS_ACCEPTED'))
        begin=dt.datetime(2026,10,1,0,0,0,tzinfo=dt.timezone.utc)
        for rd,pos,case in d.batches(config):
            directory=root/f'r{rd+1}_p{pos+1}_k{config["tasks"][case]["shape"][2]}'
            directory.mkdir()
            suite.atomic_json(directory/'worker_complete.json',dict(complete=True))
            suite.atomic_json(directory/'recovery.json',{'start_sample':{'temperature.gpu':53}})
            suite.atomic_json(directory/'telemetry_metadata.json',dict(timezone_offset_seconds=0))
            lines=[]
            for i in range(1280):
                stamp=begin+dt.timedelta(seconds=i*.05)
                fields=[stamp.strftime('%Y/%m/%d %H:%M:%S.%f')[:-3],gpu,'P0','53','99',
                        '705' if case==0 else '720','6801','69.9' if case==0 else '64',
                        '70','0x4' if case==0 else '0x0','Active' if case==0 else 'Not Active',
                        'Not Active','Not Active','0','0','0']
                lines.append(','.join(fields))
            (directory/'telemetry.csv').write_text('\n'.join(lines)+'\n')
            for trial in range(16):
                at=begin+dt.timedelta(seconds=trial*4)
                block=lambda a,b:dict(host_envelope_start_utc=(at+dt.timedelta(seconds=a)).isoformat(),
                                     host_envelope_end_utc=(at+dt.timedelta(seconds=b)).isoformat())
                row=dict(task=config['tasks'][case],trial=trial,condition='recovered_first' if trial==0 else 'continued',
                         start_utc=at.isoformat(),end_utc=(at+dt.timedelta(seconds=3.8)).isoformat(),
                         correctness={'passed':True},timing=dict(median_ms=.1,flags=[],
                         setup=dict(settle_blocks=[block(0,3)],calibration=[block(3,3.1)]),
                         blocks=[block(3.1+i*.1,3.2+i*.1) for i in range(7)]))
                suite.atomic_json(directory/f'trial_{trial:02d}.json',row)
        for tool in ('ncu','nsys'):
            for k in (72,127):
                directory=root/f'{tool}_k{k}'
                directory.mkdir()
                suite.atomic_json(directory/'profile_status.json',dict(status='COMPLETE'))
        summary=d.analyze(root,loaded,config)
        assert summary['recorded']==128 and summary['missing']==0 and summary['profiles_complete']==4
        assert summary['gate']=='REVIEW' and summary['collection']=='COMPLETE' and not summary['issues']
        assert all(r['timing']['sw_power_cap_samples']>0 if r['k']==72 else r['timing']['sw_power_cap_samples']==0
                   for r in summary['trials'])
        assert all(abs(r['gap_from_previous_trial_ms']-200)<.001 for r in summary['trials'] if r['trial']>0)
        suite.atomic_json(root/'r1_p1_k72/recovery.json',{'start_sample':{'temperature.gpu':49}})
        (root/'ncu_k72/profile_status.json').unlink()
        summary=d.analyze(root,loaded,config)
        assert summary['gate']=='FAIL' and not summary['pair_controls'][0]['matched']
        assert summary['profiles_complete']==3 and 'Round 1: START_TEMPERATURE_UNMATCHED' in summary['issues']
    print('CHECK PASS: scope/seeds; sampled recovery and timeout; 16-trial worker; 1/20-call profile workers; telemetry faults; lock/reset failures and interrupt; profiler commands/parsers; 128-trial return analysis, mismatched starts and missing profiles; ASCII sources.')


if __name__=='__main__':
    main()
