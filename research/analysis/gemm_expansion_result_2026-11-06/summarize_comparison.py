#!/usr/bin/env python3
"""Verify completed expansion fits and report both fixed validation cohorts."""
from collections import defaultdict
import csv
from itertools import groupby
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import compare_expansion as c

OUT=c.OUT
FIELDS=['learner','target','dtype','mode','added','n','seed']


def rows(path):return list(c.csv_rows(path))
def key(row,validation=False):return tuple(str(row[k]) for k in FIELDS+(['validation'] if validation else []))
def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in data])


def main():
    p=c.read(OUT/'frozen_protocol.json');completed=c.read(OUT/'execution_completed.json')
    assert completed['configurations']==576 and completed['actual_fits']==432 and completed['saved_full_models']==144
    assert completed['frozen_inputs_unchanged'] and completed['prediction_failures']==0
    for name,digest in {**p['inputs'],**p['baseline_weights']}.items():assert c.old.sha(c.ROOT/name)==digest
    metrics=rows(OUT/'metrics.csv');fits=rows(OUT/'fit_diagnostics.csv');lookup={key(r,True):r for r in metrics}
    reference={}
    for row in c.csv_rows(c.BASE/'runs/v1/validation_predictions.csv'):
        if row['rq']=='RQ1' and row['n']=='3345':
            reference[(row['learner'],row['method'],row['dtype'],row['mode'],row['seed'],row['shape'])]=(float(row['actual_ms']),float(row['predicted_ms']))
    assert len(reference)==103104
    exact=0;recomputed=0;full_cache={};old_cache={}
    for identity,group in groupby(c.csv_rows(OUT/'validation_predictions.csv'),key=key):
        grouped=defaultdict(list)
        for row in group:
            grouped[row['validation']].append(row)
            if row['added']=='0' and row['validation']=='legacy':
                ref=tuple(row[k] for k in ['learner','target','dtype','mode','seed','shape'])
                assert reference[ref]==(float(row['actual_ms']),float(row['predicted_ms']));exact+=1
        pooled_y=[];pooled_pred=[]
        for split,rr in grouped.items():
            y=np.array([float(r['actual_ms']) for r in rr]);pred=np.array([float(r['predicted_ms']) for r in rr])
            pooled_y.extend(y);pooled_pred.extend(pred)
            scored=c.old.metrics(y,pred)
            for name,value in scored.items():
                if isinstance(value,bool):assert str(value)==lookup[identity+(split,)][name]
                else:assert value==float(lookup[identity+(split,)][name])
            recomputed+=1
            if identity[4] in {'0',str(p['added_sizes'][-1])}:
                dest=old_cache if identity[4]=='0' else full_cache
                dest[identity+(split,)]=(y,pred)
        pooled=c.old.metrics(np.asarray(pooled_y),np.asarray(pooled_pred))
        for name,value in pooled.items():
            if isinstance(value,bool):assert str(value)==lookup[identity+('pooled',)][name]
            else:assert value==float(lookup[identity+('pooled',)][name])
        recomputed+=1
    assert exact==103104 and recomputed==1728
    weight_checks=[]
    v_shapes={'legacy':p['legacy_validation_shapes'],'expansion':p['new_validation_shapes']}
    recorded_weights={r['file']:r['sha256'] for r in rows(OUT/'model_weight_verification.csv')}
    for path in sorted((OUT/'models').glob('*.joblib')):
        assert c.old.sha(path)==recorded_weights[path.name]
        e=joblib.load(path);assert e['n']==3345+p['added_sizes'][-1] and not e['locked_test_opened']
        for split,shapes in v_shapes.items():
            x,lower=c.old.design(shapes,e['dtype'])
            pred=np.exp(e['model'].predict(x))*(lower if e['target']=='normalized' else 1)
            assert np.array_equal(pred,full_cache[key(e)+(split,)][1]),path.name
        weight_checks.append(dict(file=path.name,sha256=c.old.sha(path),predictions_exact=True))
    assert len(weight_checks)==144
    c.old.save_csv(OUT/'independent_weight_verification.csv',weight_checks)
    aggregate=[]
    for identity,rr in groupby(sorted(metrics,key=lambda r:tuple(r[n] for n in FIELDS[:-1]+['validation'])),key=lambda r:tuple(r[n] for n in FIELDS[:-1]+['validation'])):
        rr=list(rr);assert len(rr)==3
        info=dict(zip(FIELDS[:-1]+['validation'],identity))
        aggregate.append({**info,'mean_mape_pct':float(np.mean([float(r['mape_pct']) for r in rr])),
            'mean_p90_pct':float(np.mean([float(r['p90_pct']) for r in rr])),
            'min_seed_mape_pct':min(float(r['mape_pct']) for r in rr),'max_seed_mape_pct':max(float(r['mape_pct']) for r in rr),
            'failures':sum(int(r['failures']) for r in rr)})
    c.old.save_csv(OUT/'aggregate_metrics.csv',aggregate)
    selected=[]
    for endpoint in [0,p['added_sizes'][-1]]:
        for dtype in c.old.DTYPES:
            for mode in c.old.MODES:
                cc=[r for r in aggregate if r['added']==str(endpoint) and r['dtype']==dtype and r['mode']==mode and r['validation']=='pooled' and not r['failures']]
                selected.append(min(cc,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['learner'],r['target'])))
    c.old.save_csv(OUT/'selected_full_endpoints.csv',selected)
    accuracy=[]
    for dtype in c.old.DTYPES:
        for mode in c.old.MODES:
            cc=[r for r in aggregate if r['dtype']==dtype and r['mode']==mode and r['validation']=='pooled' and not r['failures']]
            accuracy.append(min(cc,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],int(r['n']),r['learner'],r['target'])))
    c.old.save_csv(OUT/'accuracy_first_all_sizes.csv',accuracy)
    replay_count=save_accuracy_weights(accuracy,p)
    c.old.save_json(OUT/'selection_frozen.json',dict(status='DEVELOPMENT_SELECTION_FROZEN',time=c.old.stamp(),
        protocol_sha256=c.old.sha(OUT/'frozen_protocol.json'),selected_sha256=c.old.sha(OUT/'selected_full_endpoints.csv'),
        selected_full_model_count=6,selected_baseline_model_count=6,accuracy_first_sha256=c.old.sha(OUT/'accuracy_first_all_sizes.csv'),
        accuracy_selection_policy_sha256=c.old.sha(c.HERE/'accuracy_selection_policy.json'),locked_test_opened=False))
    deltas=[];group_errors=[];rng=np.random.default_rng(20261106)
    for chosen in selected:
        if chosen['added']=='0':continue
        for split in ['legacy','expansion','pooled']:
            changes=[];after_errors=[]
            for seed in c.old.SEEDS:
                full={**chosen,'seed':seed};before={**full,'added':0,'n':3345}
                if split=='pooled':
                    a=[old_cache[key(before)+(s,)] for s in ['legacy','expansion']]
                    b=[full_cache[key(full)+(s,)] for s in ['legacy','expansion']]
                    y=np.concatenate([v[0] for v in a]);pa=np.concatenate([v[1] for v in a]);pb=np.concatenate([v[1] for v in b])
                else:
                    y,pa=old_cache[key(before)+(split,)];yb,pb=full_cache[key(full)+(split,)];assert np.array_equal(y,yb)
                changes.append((abs(pb-y)/y-abs(pa-y)/y)*100)
                after_errors.append(abs(pb-y)/y*100)
            mean_group_change=np.mean(changes,axis=0)
            samples=mean_group_change[rng.integers(0,len(y),size=(2000,len(y)))].mean(axis=1)
            deltas.append(dict(learner=chosen['learner'],target=chosen['target'],dtype=chosen['dtype'],mode=chosen['mode'],validation=split,
                delta_mape_pp=float(mean_group_change.mean()),ci95_low_pp=float(np.percentile(samples,2.5)),ci95_high_pp=float(np.percentile(samples,97.5)),
                groups=len(y),interpretation='Group bootstrap after development model selection, conditional on fixed before/after models.'))
            if split!='pooled':
                for shape,error,change in zip(v_shapes[split],np.mean(after_errors,axis=0),mean_group_change):
                    group_errors.append(dict(dtype=chosen['dtype'],mode=chosen['mode'],learner=chosen['learner'],target=chosen['target'],
                        validation=split,shape=c.old.shape_id(shape),mean_seed_ape_pct=float(error),delta_mape_pp=float(change)))
    c.old.save_csv(OUT/'selected_paired_group_bootstrap.csv',deltas)
    c.old.save_csv(OUT/'selected_group_errors.csv',sorted(group_errors,key=lambda r:-r['mean_seed_ape_pct']))
    endpoint=[]
    for added in [0,p['added_sizes'][-1]]:
        for family in c.learners.LEARNERS:
            for target in c.learners.TARGETS:
                for dtype in c.old.DTYPES:
                    for split in ['legacy','expansion','pooled']:
                        rr=[r for r in aggregate if (r['added'],r['learner'],r['target'],r['dtype'],r['validation'])==(str(added),family,target,dtype,split)]
                        assert len(rr)==3
                        endpoint.append(dict(added=added,n=3345+added,learner=family,target=target,dtype=dtype,validation=split,
                            mean_mode_mape_pct=float(np.mean([r['mean_mape_pct'] for r in rr])),
                            mean_mode_p90_pct=float(np.mean([r['mean_p90_pct'] for r in rr]))))
    c.old.save_csv(OUT/'endpoint_summary.csv',endpoint)
    c.old.save_json(OUT/'independent_verification.json',dict(status='PASS',metrics_recomputed=recomputed,
        old_endpoint_predictions_exact=exact,full_models_exact=144,validation_prediction_rows=576*(716+len(p['new_validation_shapes'])),
        accuracy_selection_replay_fits=replay_count,
        frozen_inputs_unchanged=True,locked_test_opened=False))
    plot(aggregate)
    report(aggregate,selected,endpoint,deltas,completed,accuracy,replay_count)
    print('PASS: 103104 original predictions, 1728 metrics and 144 full models verified; report and figures saved',flush=True)


def save_accuracy_weights(selected,p):
    partial=[r for r in selected if r['added'] not in {'0',str(p['added_sizes'][-1])}]
    wanted={key({**r,'seed':seed}) for r in partial for seed in c.old.SEEDS}
    expected=defaultdict(list)
    if wanted:
        for row in c.csv_rows(OUT/'validation_predictions.csv'):
            identity=key(row)
            if identity in wanted:expected[identity+(row['validation'],)].append(float(row['predicted_ms']))
        train,validation,access=c.source_labels(p)
        c.old.save_json(OUT/'accuracy_selection_label_access_log.json',access)
        (OUT/'accuracy_models').mkdir(exist_ok=True)
    refs=[];replays=[]
    for chosen in selected:
        for seed in c.old.SEEDS:
            info={k:int(chosen[k]) if k in {'added','n'} else chosen[k] for k in FIELDS[:-1]};info['seed']=seed
            name=f"{chosen['learner']}_{chosen['dtype']}_{chosen['mode']}_{chosen['target']}_{seed}.joblib"
            if chosen['added']=='0':
                path=c.BASE/'runs/v1/models'/f"rq1_{chosen['learner']}_{chosen['dtype']}_{chosen['mode']}_{chosen['target']}_3345_{seed}.joblib"
            elif chosen['added']==str(p['added_sizes'][-1]):path=OUT/'models'/name
            else:
                shapes=p['old_train_orders'][str(seed)]+p['new_train_orders'][str(seed)][:int(chosen['added'])]
                x,lower=c.old.design(shapes,chosen['dtype']);y=c.old.target(train,shapes,chosen['dtype'],chosen['mode'])
                response=np.log(y)-(np.log(lower) if chosen['target']=='normalized' else 0)
                model,detail=c.learners.fit(chosen['learner'],x,response,seed)
                for split,ss in [('legacy',p['legacy_validation_shapes']),('expansion',p['new_validation_shapes'])]:
                    xv,lv=c.old.design(ss,chosen['dtype']);pred=np.exp(model.predict(xv))*(lv if chosen['target']=='normalized' else 1)
                    assert np.array_equal(pred,np.asarray(expected[key(info)+(split,)])),'Intermediate selected model replay differs'
                path=OUT/'accuracy_models'/name
                joblib.dump(dict(model=model,**info,features=c.old.FEATURES,protocol_sha256=c.old.sha(OUT/'frozen_protocol.json'),
                    scope='RQ1 full legacy_train including original C; preserve A/C separation for RQ4',
                    validation_only=True,locked_test_opened=False),path)
                replays.append({**info,**detail,'predictions_exact':True})
            refs.append({**info,'model_ref':str(path.relative_to(c.ROOT)),'sha256':c.old.sha(path)})
    c.old.save_csv(OUT/'accuracy_first_model_refs.csv',refs)
    if replays:c.old.save_csv(OUT/'accuracy_selection_replay_diagnostics.csv',replays)
    return len(replays)


def plot(aggregate):
    old_aggregate=rows(c.BASE/'runs/v1/aggregate_metrics.csv')
    colors=dict(rf='#1f67a1',ridge='#c56524',mlp32='#25835c',mlp64='#994c99')
    for split in ['legacy','expansion']:
        for measure in ['mape','p90']:
            fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
            for row,dtype in enumerate(c.old.DTYPES):
                for col,mode in enumerate(c.old.MODES):
                    ax=axes[row,col]
                    for family in c.learners.LEARNERS:
                        for target,style in [('direct','-'),('normalized','--')]:
                            rr=[(int(r['n']),float(r[f'mean_{measure}_pct'])) for r in aggregate if (r['learner'],r['target'],r['dtype'],r['mode'],r['validation'])==(family,target,dtype,mode,split)]
                            if split=='legacy':
                                rr += [(int(r['n']),float(r[f'mean_{measure}_pct'])) for r in old_aggregate if (r['rq'],r['learner'],r['method'],r['dtype'],r['mode'])==('RQ1',family,target,dtype,mode) and int(r['n'])<3345]
                            rr=sorted(rr)
                            ax.plot([x for x,_ in rr],[y for _,y in rr],style,marker='o',markersize=3,color=colors[family],label=family.upper()+' '+('direct' if target=='direct' else 'norm'))
                    ax.set_xscale('log');ax.set_yscale('log');ax.set(title=f'{dtype.upper()} / {mode}',xlabel='Training MNK shapes',ylabel=f'Mean validation {measure.upper()} (%)')
                    ax.grid(alpha=.2,which='both')
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
            fig.suptitle(f'GEMM expansion: {split} validation, 3 seeds; fixed learner settings')
            for suffix in ['png','pdf']:fig.savefig(OUT/f'{split}_{measure}_learning_curve.{suffix}',dpi=170)
            plt.close(fig)


def report(aggregate,selected,endpoint,deltas,completed,accuracy,replay_count):
    p=c.read(OUT/'frozen_protocol.json');dataset=c.read(OUT/'dataset_summary.json');verified=c.read(c.HERE/'verification.json')
    full_n=dataset['augmented_train_shapes'];new_n=dataset['common_shape_counts']['expansion_train'];val_n=dataset['common_shape_counts']['expansion_validation']
    def metric(learner,target,dtype,mode,added,split):
        return next(r for r in aggregate if (r['learner'],r['target'],r['dtype'],r['mode'],r['added'],r['validation'])==(learner,target,dtype,mode,str(added),split))
    full=[r for r in selected if r['added']!='0'];before=[r for r in selected if r['added']=='0']
    pairs=[]
    for r in full:
        vals=[]
        for split in ['legacy','expansion','pooled']:
            a=metric(r['learner'],r['target'],r['dtype'],r['mode'],0,split);b=metric(r['learner'],r['target'],r['dtype'],r['mode'],new_n,split)
            vals.append(f"{a['mean_mape_pct']:.3f} → {b['mean_mape_pct']:.3f}")
        pairs.append([r['dtype'],r['mode'],r['learner']+' '+r['target']]+vals)
    text=['# 추가 GEMM 10,000형상: 검산과 자료 보강 효과','',
        f'2026-11-06 사용자 반환 ZIP을 검산하고 기존 RF·Ridge·MLP32/64 설정으로 학습을 완료했다. 공통 품질 조건을 통과한 새 학습 {new_n:,}형상과 기존 3,345형상을 합쳐 {full_n:,}형상을 사용했다. 평가 구역은 기존 716형상과 새 {val_n:,}형상이다. 이번 비교의 대상은 RQ1 GEMM 직접/정규화 예측과 자료량 효과다.','',
        '## 수집과 품질','',
        f"원본은 `expansion_10000_20261105_181927_9a759e.zip`이며 SHA-256은 `{verified['archive_sha256']}`다. 원본을 연구 폴더에 해시 일치 사본으로 보존하고 ZIP의 189,361항목 CRC와 안전한 경로, 로컬 배포 source 223개·의존성 91개를 확인했다. 고정된 10,000형상·180,000 task와 원래 split이 일치했다. 원자료 180,000건의 누락·중복은 0개이며 저장된 정확성·유한값 검사는 모두 통과했다.",'',
        f"원래 분석기로 1,500개 센서 구간·1,260,000블록을 재계산해 summary와 exports를 바이트 단위로 재현했다. 실험 판정 REVIEW를 보존했고 전체 보류는 6,041건이다. 개별 판정은 PASS 174,000건·REVIEW 6,000건이며 회차 간 기준을 넘은 14조건의 추가 보류가 반영됐다. 회차 간 범위는 원래 5% 기준을 유지했다. 전체 정상 사용 가능 원측정은 {verified['accepted_measurements']:,}건이다.",'',
        f"센서 {verified['sensor_samples']:,}표본의 관측 범위는 `{json.dumps(verified['sensor_ranges'],ensure_ascii=False)}`다. 보류 구간 사유와 빈도는 [검산 JSON]({c.HERE/'verification.json'})에 남겼다. 누적 제한 카운터는 기존 미해결 항목으로 유지한다.",'',
        f"Worker 수집 시간은 {verified['worker_elapsed_seconds']:,.2f}초, 약 {verified['worker_elapsed_seconds']/86400:.3f}일이다. 개별 task wall 합은 {verified['task_wall_sum_seconds']:,.2f}초, 중앙값은 {verified['task_wall_median_seconds']:.3f}초다. 수집 완료 시각은 UTC `{verified['finished_utc']}`이며 한국 시간 2026-11-05 18:15경이다. 이 비용은 보류 건도 포함한 실제 수집이다.",'',
        '## 공통 학습 자료','',
        '품질 보류 task를 먼저 제외하고, 같은 M/N/K의 두 dtype·세 mode에서 세 회차가 모두 정상인 경우를 공통 형상으로 채택했다. 네 학습기는 같은 형상과 정답을 공유한다. 부분적으로 정상인 형상의 원측정과 원래 품질 이력은 원본 및 eligibility 표에 보존한다.','',
        table(['구역','계획 형상','공통 채택 형상','비교에서 보류한 형상'],[
            ['새 train',9000,new_n,9000-new_n],['새 validation',1000,val_n,1000-val_n],['기존 train',3345,3345,0],['기존 validation',716,716,0]]),'',
        f"새 공통 형상 {new_n+val_n:,}개는 {dataset['common_conditions']:,}조건, {dataset['common_raw_records']:,}원측정에 대응한다. 이 수량과 개별 사용 가능 {verified['accepted_measurements']:,}원측정을 구분한다. 모델 하나의 학습 표본 수는 {full_n:,}개이고, 두 dtype·세 mode에 걸친 학습 조건은 {full_n*6:,}개다. 학습 자료는 검증 구역과 고유 M/N/K가 겹치지 않는다.",'',
        '## 같은 설정에서 자료량만 변경','',
        f'기존 3,345형상 가중치 144개를 재사용하고, 새 형상 1,000개·3,000개·{new_n:,}개를 각각 추가한 중첩 학습을 수행했다. 고유 학습량은 3,345 / 4,345 / 6,345 / {full_n:,}이다. 네 학습기 × 두 목표 표현 × 두 dtype × 세 mode × 세 seed × 네 자료량의 576구성이며 새 fit은 432회다. 기존 endpoint를 같은 두 검증 구역에서 다시 평가했다.','',
        '공통 입력은 원래 15개 정적 특징이며 직접 목표 log(T)와 경험적 Roofline 정규화 log(T/L)를 각각 학습했다. RF는 100 trees/depth 8/leaf 2, Ridge는 alpha=1, MLP는 (32,32)/(64,64)·ReLU·Adam·batch 최대 128·최대 3,000 epoch다. MLP의 입력과 log 목표 표준화는 학습 표본 내부에서만 fit했고 학습 손실 기반 patience 100으로 종료했다. 정밀도별 경험적 상한과 bandwidth 상수도 기존 값 그대로다.','',
        '추가 자료의 정답을 읽기 전 prelabel protocol에 자료 채택 규칙·비교 격자·오차 우선 선택을 고정했다. 품질 검산 후 실제 채택 ID·코드·입력·기존 가중치 해시를 별도 frozen protocol에 저장했다. 기존 A 다음 C 순서에 새 형상을 seed별 시간값 없는 hash 순서로 덧붙였다. 전체 legacy_train을 사용하는 이번 RQ1 모델과, source A/C를 분리하는 RQ4 protocol의 역할을 유지한다.','',
        '## 전체 보강 endpoint의 선택','',
        f'기존 716개와 새 {val_n}개를 합친 {716+val_n}검증 그룹의 pooled MAPE를 세 seed에서 평균해 학습기·목표 표현을 선택했다. 같은 seed별 pooled p90을 평균해 함께 기록했다. 개별 구역의 MAPE·p90은 별도 표와 CSV에 남겼다.','',
        table(['dtype','mode','학습기','목표','보강 전 최선 MAPE','보강 후 MAPE','보강 후 p90'],[
            [r['dtype'],r['mode'],r['learner'],r['target'],f"{next(a for a in before if a['dtype']==r['dtype'] and a['mode']==r['mode'])['mean_mape_pct']:.3f}%",f"{r['mean_mape_pct']:.3f}%",f"{r['mean_p90_pct']:.3f}%"] for r in full]),'',
        '각 endpoint 안에서 고른 최선끼리의 비교다. 아래 표는 보강 후 선택한 학습기·목표를 보강 전에도 똑같이 적용해, 자료량 변화 자체를 비교한다.','',
        table(['dtype','mode','고정 학습기·목표','기존 검증 MAPE 전→후','새 검증 MAPE 전→후','합친 검증 MAPE 전→후'],pairs),'',
        '## 오차 최소화 우선의 전체 격자 선택','',
        '자료량 효과를 보는 주 비교는 위의 전체 보강 endpoint다. 사용자 선호에 따른 현재 가중치 추천은 같은 네 자료량과 학습기·목표 격자 전체에서 pooled MAPE를 최소화해 별도로 정리했다. 여섯 조건 모두 전체 보강 모델이 선택됐다. 선택 규칙은 주 학습 중 별도 파일로 기록했다.','',
        table(['dtype','mode','학습기','목표','학습 형상 수','MAPE','p90'],[
            [r['dtype'],r['mode'],r['learner'],r['target'],r['n'],f"{r['mean_mape_pct']:.3f}%",f"{r['mean_p90_pct']:.3f}%"] for r in accuracy]),'',
        f'선택 가중치 참조는 18개이며 추가 가중치 재현 fit은 {replay_count}회다. 선택된 전체 보강 가중치의 경로와 해시를 참조 표에 고정했다.','',
        '## 학습기별 전체 endpoint','']
    for split in ['legacy','expansion','pooled']:
        data=[]
        for family in c.learners.LEARNERS:
            for target in c.learners.TARGETS:
                vals=[]
                for dtype in c.old.DTYPES:
                    a=next(r for r in endpoint if (r['added'],r['learner'],r['target'],r['dtype'],r['validation'])==(0,family,target,dtype,split))
                    b=next(r for r in endpoint if (r['added'],r['learner'],r['target'],r['dtype'],r['validation'])==(new_n,family,target,dtype,split))
                    vals.append(f"{a['mean_mode_mape_pct']:.3f} → {b['mean_mode_mape_pct']:.3f}")
                data.append([family,target]+vals)
        text += [f'### {split} validation','',table(['학습기','목표','FP16 세 mode 평균 MAPE (%)','FP32 세 mode 평균 MAPE (%)'],data),'']
    text += ['## 학습곡선','',
        '점은 세 seed 평균이며 실선은 직접 목표, 점선은 정규화 목표다. 가로축은 고유 학습 M/N/K 수다. 기존 validation 그림은 보존된 작은 학습량의 결과와 연결했고 새 validation은 이번에 동일하게 평가한 네 자료량을 표시했다.','']
    for split in ['legacy','expansion']:
        for measure in ['mape','p90']:text += [f"![{split} {measure} learning curve]({OUT/f'{split}_{measure}_learning_curve.png'})",'']
    fit_rows=rows(OUT/'fit_diagnostics.csv');diagnostics=[]
    for family in c.learners.LEARNERS:
        ff=[r for r in fit_rows if r['learner']==family and r['added']==str(new_n)]
        vv=[r for r in aggregate if r['learner']==family and r['added']==str(new_n) and r['validation']=='pooled']
        epochs=[int(r['epochs']) for r in ff if r['epochs']]
        diagnostics.append([family,f"{np.mean([float(r['train_mape_pct']) for r in ff]):.3f}",
            f"{np.mean([r['mean_mape_pct'] for r in vv]):.3f}",
            f'{min(epochs)}~{max(epochs)}' if epochs else '-',sum(int(r['convergence_warnings']) for r in ff)])
    text += ['이번 자료 확대에서는 신경망의 개선 폭이 컸다. FP16의 현재 최저 구성은 MLP32로 바뀌었고 FP32는 RF 정규화가 가장 낮았다. Ridge FP16은 기존 검증 구역의 오차가 커졌다. 자료 확대의 효과는 학습기와 검증 구역에 따라 달랐다.','',
        '## 전체 보강 모델의 학습 상태','',
        table(['학습기','평균 train MAPE (%)','평균 pooled validation MAPE (%)','MLP epoch 범위','수렴 경고'],diagnostics),'',
        '각 값은 두 목표 표현·두 dtype·세 mode·세 seed를 같은 비중으로 평균했다. 위의 최저 오차 구성 선택과 별도로 학습기 전체의 학습·검증 상태를 보여 준다.','',
        '## 검산과 비용','',
        f"원래 3,345 endpoint의 validation 예측 103,104개를 원소별 정확히 재현했다. 1,728개 조건별 지표를 저장 예측에서 다시 계산했고 전체 보강 모델 144개의 두 검증 구역 예측도 직렬화 후 정확히 일치했다. 새 fit의 수렴 경고는 {completed['convergence_warning_fits']}개다. 실제 비교 wall은 {completed['wall_seconds']:.2f}초, fit 합은 {completed['total_fit_seconds']:.2f}초다. 무효 예측은 0개다.",'',
        '네 학습기가 정답을 공유하므로 수집 비용을 모델 수만큼 곱하지 않는다. 같은 validation을 유지한 before/after 비교와 그룹 단위 paired bootstrap을 함께 보존했다. 보류 6,041개는 재측정 대상으로 남아 있고, 현재 표는 공통 품질을 통과한 개발 검증 구역의 결과다. 최종 시험은 잠금 상태를 유지한다.','',
        '## 파일','',
        f"- [원자료 검산]({c.HERE/'verification.json'}) / [검산 코드]({c.HERE/'verify_expansion.py'})",
        f"- [자료 채택 집계]({OUT/'dataset_summary.json'}) / [형상별 채택 여부]({OUT/'shape_eligibility.csv'})",
        f"- [정답 열람 전 설계]({c.HERE/'prelabel_comparison_protocol.json'}) / [최종 protocol]({OUT/'frozen_protocol.json'})",
        f"- [조건별 집계]({OUT/'aggregate_metrics.csv'}) / [전체 보강 endpoint 선택]({OUT/'selected_full_endpoints.csv'})",
        f"- [오차 우선의 전체 격자 선택]({OUT/'accuracy_first_all_sizes.csv'}) / [선택 가중치]({OUT/'accuracy_first_model_refs.csv'})",
        f"- [전체 보강 선택 모델의 형상별 오차]({OUT/'selected_group_errors.csv'}) / [세부 구간 지표]({OUT/'subgroup_metrics.csv'})",
        f"- [독립 재현 검산]({OUT/'independent_verification.json'}) / [paired bootstrap]({OUT/'selected_paired_group_bootstrap.csv'})",'']
    (c.HERE/'추가GEMM_10000형상_분석결과_2026-11-06.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
