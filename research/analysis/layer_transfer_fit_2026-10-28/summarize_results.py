#!/usr/bin/env python3
"""Verify and summarize completed RQ3 fits; read RQ2 only after frozen selection."""
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import compare_layers as c

OUT=c.layer.HERE/'runs/matched_v1'
RESULT=c.layer.ROOT/'analysis/layer_transfer_core_dev_result_2026-10-28'
RAW=RESULT/'raw/core_dev_20261028_143530_VEnIY5'


def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in data])


def key(r,group=False):
    fields=['layer','dtype','method','k','seed']+(['group_id'] if group else [])
    return tuple(str(r[k]) for k in fields)


def independent_prior(groups,kind,dtype,source,target):
    parts=[[(g['M'],g['F'],g['D']) for g in groups]]
    if kind=='ffn':parts.append([(g['M'],g['D'],g['F']) for g in groups])
    values=[]
    # Keep the measured prediction batch sizes; MLP BLAS rounding can change with batching.
    for shapes in parts:
        x,lower=c.gemm.design(shapes,dtype)
        values.append(np.exp(source.predict(x))*(lower if target=='normalized' else 1))
    return values[0]+values[1] if kind=='ffn' else values[0]


def main():
    complete=c.layer.read(OUT/'selection_frozen.json');p=c.layer.read(OUT/'frozen_protocol.json')
    assert complete['status']=='RQ3_SELECTION_FROZEN_VALIDATION_ONLY' and complete['explicit_values_read']==0
    assert c.gemm.sha(OUT/'selected_configurations.csv')==complete['selected_configurations_sha256']
    assert c.layer.read(RESULT/'verification_all.json')['status']=='PASS'
    for name,digest in {**p['inputs'],**p['source_weights']}.items():assert c.gemm.sha(c.layer.ROOT/name)==digest
    predictions=rows(OUT/'validation_predictions.csv');scores=rows(OUT/'metrics.csv')
    aggregate=rows(OUT/'aggregate_metrics.csv');selected=rows(OUT/'selected_configurations.csv')
    diagnostic=rows(OUT/'fit_diagnostics.csv');original=c.layer.read(c.layer.HERE/'frozen_fit_protocol.json')
    assert len(scores)==960 and len(predictions)==9600 and len(aggregate)==320
    assert sum(int(r['failures']) for r in scores)==0
    baseline={key(r,True):(float(r['actual_ms']),float(r['predicted_ms'])) for r in rows(c.layer.HERE/'runs/dev_fit_v1/validation_predictions.csv')}
    matched=0;by_config=defaultdict(list)
    for row in predictions:
        by_config[(row['learner'],)+key(row)].append(row)
        if row['learner']=='rf':
            assert baseline[key(row,True)]==(float(row['actual_ms']),float(row['predicted_ms']))
            matched+=1
    assert matched==len(baseline)==2400
    for row in scores:
        rr=by_config[(row['learner'],)+key(row)]
        metric=c.gemm.metrics(np.array([float(r['actual_ms']) for r in rr]),np.array([float(r['predicted_ms']) for r in rr]))
        for name,value in metric.items():
            if isinstance(value,bool):assert str(value)==row[name]
            else:assert float(row[name])==value
    weight_checks=[]
    vg=[original['groups'][i] for i in original['validation_ids']]
    for path in sorted((OUT/'models').glob('*.joblib')):
        e=joblib.load(path);ref=e['source_ref'];assert c.gemm.sha(c.layer.ROOT/ref)==e['source_sha256']==p['source_weights'][ref]
        source=joblib.load(c.layer.ROOT/ref)['model']
        prior=independent_prior(vg,e['layer'],e['dtype'],source,e['source_target'])
        x=c.layer.features(vg,e['layer'],e['dtype']);method=e['method']
        if method=='zero':prediction=prior
        elif method=='constant_ratio':prediction=prior*np.exp(e['model'])
        else:
            xx=np.column_stack((x,np.log(prior))) if method in {'residual','prior_direct'} else x
            prediction=np.exp(e['model'].predict(xx))*(prior if method=='residual' else 1)
        identity=(e['learner'],)+key(e)
        expected=np.array([float(r['predicted_ms']) for r in by_config[identity]])
        assert np.array_equal(prediction,expected),path.name
        weight_checks.append(dict(file=path.name,sha256=c.gemm.sha(path),exact_predictions=len(prediction)))
    assert len(weight_checks)==87 and sum(int(r['convergence_warnings']) for r in diagnostic)==0
    c.gemm.save_csv(OUT/'independent_weight_verification.csv',weight_checks)
    verification=dict(status='PASS',metrics_recomputed=960,validation_predictions=9600,original_rf_predictions_exact=matched,
        saved_models_independently_exact=len(weight_checks),convergence_warnings=0,invalid_predictions=0,
        source_hashes_unchanged=True,locked_test_opened=False)
    c.gemm.save_json(OUT/'independent_verification.json',verification)
    best=[r for r in selected if r['selection_scope']=='all' and r['selection_rule']=='accuracy_first']
    assert len(best)==4
    rng=np.random.default_rng(20261028);uncertainty=[];worst=[]
    for chosen in best:
        rr=[r for r in predictions if all(r[n]==chosen[n] for n in ['learner','layer','dtype','method','k'])]
        ids=sorted({r['group_id'] for r in rr});assert len(ids)==10 and len(rr)==30
        errors=[]
        for group_id in ids:
            group=[r for r in rr if r['group_id']==group_id]
            errors.append(np.mean([abs(float(r['predicted_ms'])/float(r['actual_ms'])-1)*100 for r in group]))
            worst.append(dict(layer=chosen['layer'],dtype=chosen['dtype'],group_id=group_id,
                mean_seed_ape_pct=float(errors[-1]),actual_ms=float(group[0]['actual_ms']),**{k:original['groups'][group_id][k] for k in ['M','D','F']}))
        samples=np.asarray(errors)[rng.integers(0,10,size=(10000,10))].mean(axis=1)
        per_seed=[float(r['mape_pct']) for r in scores if all(r[n]==chosen[n] for n in ['learner','layer','dtype','method','k'])]
        uncertainty.append(dict(layer=chosen['layer'],dtype=chosen['dtype'],selected_learner=chosen['learner'],method=chosen['method'],k=chosen['k'],
            mape_pct=float(np.mean(errors)),group_bootstrap_ci95_low=float(np.percentile(samples,2.5)),
            group_bootstrap_ci95_high=float(np.percentile(samples,97.5)),seed_mape_min=min(per_seed),seed_mape_max=max(per_seed),
            note='Development group bootstrap conditional on selected models; not post-selection-corrected or final-test confidence.'))
    c.gemm.save_csv(OUT/'selected_development_uncertainty.csv',uncertainty)
    c.gemm.save_csv(OUT/'selected_group_errors.csv',sorted(worst,key=lambda r:-r['mean_seed_ape_pct']))
    access=c.layer.read(RESULT/'explicit_diagnostics_access.json')
    assert datetime.fromisoformat(access['opened_utc'])>datetime.fromisoformat(complete['timestamp'])
    comparisons=c.layer.read(RAW/'restricted_explicit/comparison.json')['comparisons']
    assert len(comparisons)==160 and all(r['status']=='PASS' for r in comparisons)
    gaps=[]
    for kind in ['linear','ffn']:
        for dtype in ['fp16','fp32']:
            cc=[r for r in comparisons if r['layer']==kind and r['dtype']==dtype]
            ratios=np.array([r['ratio'] for r in cc])
            gaps.append(dict(layer=kind,dtype=dtype,conditions=len(cc),median_ratio=float(np.median(ratios)),
                min_ratio=float(ratios.min()),max_ratio=float(ratios.max()),
                decomposed_faster_count=int((ratios<1).sum()),standard_faster_count=int((ratios>1).sum()),
                within_5pct_count=int((abs(ratios-1)<=.05).sum()),
                median_standard_us=float(np.median([r['standard_ms'] for r in cc])*1000),
                median_decomposed_us=float(np.median([r['decomposed_ms'] for r in cc])*1000)))
    c.gemm.save_csv(RESULT/'implementation_group_summary.csv',gaps)
    standard=[c.layer.read(path) for path in sorted((RAW/'records').glob('*.json'))]
    explicit=[c.layer.read(path) for path in sorted((RAW/'restricted_explicit/records').glob('*.json'))]
    def timed_seconds(records):
        return sum(r['timing']['setup']['prepare_wall_ms']+sum(b['host_wall_ms'] for b in r['timing']['blocks']) for r in records)/1000
    costs=[]
    for k in c.layer.BUDGETS:
        union={i for s in c.layer.SEEDS for i in original['subsets']['seeds'][str(s)]['by_k'][str(k)]}
        costs.append(dict(k=k,per_seed_train_groups=k,validation_groups=10,per_seed_standard_records=12*(k+10),
            three_seed_train_union_groups=len(union),three_seed_union_standard_records=12*(len(union)+10)))
    selected_union_counts=[]
    for chosen in best:
        k=int(chosen['effective_k']);ids={i for s in c.layer.SEEDS for i in original['subsets']['seeds'][str(s)]['by_k'][str(k)]} if k else set()
        selected_union_counts.append(dict(layer=chosen['layer'],dtype=chosen['dtype'],k=k,train_union_groups=len(ids),
            per_seed_standard_records=3*(k+10),three_seed_union_standard_records=3*(len(ids)+10)))
    known_shared=sum(sum(r['preparation'].values()) for r in standard)/1000
    cost=dict(controlled_collection_wall_seconds=c.layer.read(RAW/'cost_ledger.json')['collection_wall_seconds'],
        observed_standard_timing_setup_and_host_blocks_seconds=timed_seconds(standard),
        observed_explicit_timing_setup_and_host_blocks_seconds=timed_seconds(explicit),
        observed_shared_pair_creation_and_before_correctness_seconds=known_shared,
        measured_standard_records=480,measured_explicit_records=480,
        selected_per_seed_distinct_condition_records=sum(r['per_seed_standard_records'] for r in selected_union_counts),
        selected_three_seed_union_distinct_condition_records=sum(r['three_seed_union_standard_records'] for r in selected_union_counts),
        bundled_collector_required_standard_records=480,
        historical_source_collection_wall_seconds='UNKNOWN',
        scope='Measured subset timing components; shared before-correctness counted once per pair. Excludes after-correctness, global preheat, monitoring and other overhead. Retrospective label budgets are not actual saved GPU time.')
    c.gemm.save_json(OUT/'cost_summary.json',cost)
    c.gemm.save_csv(OUT/'label_costs.csv',costs)
    c.gemm.save_csv(OUT/'selected_label_costs.csv',selected_union_counts)
    plot(aggregate,comparisons)
    report(best,selected,aggregate,gaps,cost,costs,uncertainty,verification)
    print('PASS: independent RF, metric and 87-model verification; RQ2 summary, costs, report and figures saved')


def plot(aggregate,comparisons):
    colors=dict(rf='#1964a3',ridge='#c06422',mlp32='#22805e',mlp64='#9b4a9c')
    for method in ['residual','direct']:
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
        for ax,(kind,dtype) in zip(axes.flat,[(l,d) for l in ['linear','ffn'] for d in ['fp16','fp32']]):
            for learner in c.learners.LEARNERS:
                rr=sorted([r for r in aggregate if (r['learner'],r['layer'],r['dtype'],r['method'])==(learner,kind,dtype,method)],key=lambda r:int(r['k']))
                ax.plot([int(r['k']) for r in rr],[float(r['mean_mape_pct']) for r in rr],marker='o',color=colors[learner],label=learner.upper())
            ax.set(title=f'{kind.upper()} / {dtype.upper()}',xticks=c.layer.BUDGETS,xlabel='Training layer groups (k)',ylabel='Mean validation MAPE (%)')
            ax.set_yscale('log');ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
        fig.suptitle(f'RQ3 {method}: same subsets, 3 seeds, 10 validation groups')
        for suffix in ['png','pdf']:fig.savefig(OUT/f'{method}_learning_curve.{suffix}',dpi=170)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    groups=[(l,d) for l in ['linear','ffn'] for d in ['fp16','fp32']]
    data=[[r['ratio'] for r in comparisons if (r['layer'],r['dtype'])==(l,d)] for l,d in groups]
    ax.boxplot(data,tick_labels=[f'{l.upper()}\n{d.upper()}' for l,d in groups],showfliers=False)
    random=np.random.default_rng(20261028)
    for i,values in enumerate(data,1):ax.scatter(i+random.uniform(-.1,.1,len(values)),values,s=13,alpha=.65)
    ax.axhline(1,color='black',linewidth=1,linestyle='--');ax.set(ylabel='Decomposed time / standard time',title='RQ2 development implementation gap (40 groups per condition)')
    ax.grid(axis='y',alpha=.2)
    for suffix in ['png','pdf']:fig.savefig(RESULT/f'implementation_ratios.{suffix}',dpi=170)
    plt.close(fig)


def report(best,selected,aggregate,gaps,cost,costs,uncertainty,verification):
    names=dict(zero='GEMM 합산 무보정',constant_ratio='상수 비율 보정',direct='레이어 직접 D0',prior_direct='prior 입력 직접 D+A',residual='log 잔차 보정 R')
    text=['# 새 Linear·FFN 개발 자료의 검산과 네 학습기 비교','',
        '2026-10-28 반환된 core_dev ZIP의 960측정을 검산하고 표준 레이어 자료로 RQ3 학습을 완료했다. GEMM 추가 10,000형상·180,000측정 패키지와는 별개의 결과다. 기존 잠금 시험과 새 core_locked의 시간값은 열지 않았다.','',
        '## 반환 자료와 품질','',
        '원본 ZIP은 `core_dev_20261028_143530_VEnIY5_17236749.zip`이며 SHA-256은 `5e32c060c2cd477c5675e1048b065b373df878bf9cbb20099c137cac86a3ec0f`다. 원본과 해제 자료를 보존했고 ZIP CRC, 결과 파일 해시 2,255개, 측정 source 135개가 일치했다. 실제 40그룹(train 30/validation 10) × 두 레이어 × 두 dtype × 두 구현 × 세 회차의 960측정이 모두 PASS이며 누락은 0개다.','',
        '원시 event 통계, 7블록·3초 settling·반복 수·출력 배치와 입력 seed, 저장된 정확성 JSON 960개·입력 identity 160개를 확인했다. 센서 74,835표본에서 SM 720MHz·메모리 6801MHz가 유지됐고 전력·열 제한 활성 표본은 0개, 온도는 33~61°C였다. 전체 6,720개 본측정 블록의 센서 창과 32개 원래 전체 검색 창을 대조했다. public summary와 explicit 비교 JSON/CSV도 별도 폴더에서 정확히 재현했다. 저장된 GPU 경로 일치·유한값 검사 및 표본 CPU FP64 참조를 검산했으며 새 GPU 순전파를 다시 실행한 것은 아니다.','',
        f"통제된 수집·클럭 설정/해제·분석 wall은 {cost['controlled_collection_wall_seconds']:.2f}초({cost['controlled_collection_wall_seconds']/60:.2f}분)다. 센서 요청 간격은 50ms, 관측 중앙값 52ms·최대 118ms이며 센서 자체 물리적 해상도 검증과 구분한다. 환경은 RTX 4000 SFF Ada, PyTorch 2.14.0+cu132, CUDA runtime 13.2, CUDA_HOME toolkit 13.4다.",'',
        '## 학습 방법과 정보 경계','',
        'GPU 수집 전 동결된 RF protocol과 source 가중치 6개를 이용해 원래 240구성을 먼저 실행했다. 사용자 지시의 네 학습기 비교를 레이어에도 적용하기 위해 별도 compare_layers protocol을 실제 시간값 접근 전에 동결했다. 후자는 GPU 수집 이후, 레이어 정답 열람 이전에 고정한 확장 비교이며 수집 전 사전등록으로 소급하지 않는다. 원래 RF protocol과 결과는 바꾸지 않았다.','',
        '네 학습기 RF, Ridge(alpha=1), MLP(32,32), MLP(64,64)는 같은 train 30그룹의 중첩 k=5/10/20/30, 같은 세 seed와 validation 10그룹을 사용했다. 두 레이어·두 dtype마다 별도 모델이며 세 회차의 중앙값을 정답으로 삼았다. 네 계열 × 다섯 방법 × 네 k × 세 seed × 네 layer/dtype의 960구성이고 실제 회귀기 fit은 576회다. 나머지 384구성은 무보정 또는 상수 보정이며 이 960구성은 새 GPU 측정 수와 다른 단위다.','',
        'source는 기존 A 2,652형상에서 학습한 allocate GEMM 모델을 계열·dtype·seed에 맞춰 고정했다. source의 직접/정규화 표현도 기존 선택을 재사용했다. Linear prior는 한 GEMM 예측, FFN prior는 두 GEMM 예측의 합이다. 실측 구성 연산 시간, explicit 시간, H/C와 시험 정답은 입력에 없다. source 신경망의 미세조정은 하지 않았다.','',
        '공통 정적 특징은 기존 첫 GEMM 특징 15개와 전체 GEMM FLOPs·GELU 원소 수를 합친 17개다. 직접 D0는 log(T)를 학습한다. D+A는 같은 특징에 log(prior)를 추가해 log(T)를 학습하며, 잔차 R도 같은 18개 입력으로 log(T/prior)를 학습한다. 상수 보정은 학습 표본의 log(T/prior) 평균 하나만 fit한다. MLP의 입력과 log 목표 표준화는 해당 학습 표본만 사용했고 Adam·최대 3,000 epoch·학습 손실 기반 patience 100 설정을 기존 비교에서 재사용했다. 외부 validation으로 epoch를 선택하지 않았으며 수렴 경고는 0개다.','',
        '모든 구성의 선택은 standard 자료만으로 저장한 뒤 explicit 진단을 열었다. 기존 최소 비용 15%/30% 규칙도 남겼고 현재 추천은 사용자 선호대로 validation 평균 MAPE, p90 순으로 고른다. 아래 값은 각 seed에서 계산한 지표의 세 seed 평균이다. 평균 p90은 세 seed 예측을 합친 분위수가 아니다.','',
        '## 오차 최소화 선택','',
        table(['대상','GEMM source·보정','k','MAPE (%)','p90 (%)'],[
            [r['layer']+' '+r['dtype'],r['learner']+' · '+names[r['method']],r['effective_k'],f"{float(r['mean_mape_pct']):.3f}",f"{float(r['mean_p90_pct']):.3f}"] for r in best]),'',
        'Linear FP32의 MLP64는 새 레이어 신경망을 재학습한 모델이 아니다. 고정 MLP64 GEMM source의 합에 상수 비율 하나를 보정했다. 나머지 세 조건은 RF 잔차 모델이다. k가 커질 때 오차가 항상 줄어들지는 않았으며 Linear FP16은 k=10, FFN FP16은 k=20에서 현재 격자의 가장 낮은 값을 냈다.','',
        '### 학습기별 최저 오차','',
        table(['계열','Linear FP16','Linear FP32','FFN FP16','FFN FP32'],[
            [f]+[f"{float(r['mean_mape_pct']):.3f}% ({r['method']}, k={r['effective_k']})" for l,d in [('linear','fp16'),('linear','fp32'),('ffn','fp16'),('ffn','fp32')] for r in selected if (r['selection_scope'],r['selection_rule'],r['learner'],r['layer'],r['dtype'])==(f,'accuracy_first',f,l,d)] for f in c.learners.LEARNERS]),'',
        '각 셀은 해당 계열 내부의 개발 validation 선택이다. RF가 모든 레이어에서 최적이라고 결론내리지 않는다. Ridge FFN FP32의 최저값은 보정 없이 source를 사용하는 기준선이며 보정이 개선을 주지 않았다.','',
        '### 레이어 직접 학습과 비교','']
    direct=[]
    for chosen in best:
        cc=[r for r in aggregate if (r['layer'],r['dtype'],r['method'])==(chosen['layer'],chosen['dtype'],'direct')]
        rr=min(cc,key=lambda r:float(r['mean_mape_pct']))
        direct.append([chosen['layer']+' '+chosen['dtype'],rr['learner'],rr['k'],f"{float(rr['mean_mape_pct']):.3f}",f"{float(chosen['mean_mape_pct']):.3f}"])
    text += [table(['대상','직접 최저 계열','직접 k','직접 MAPE (%)','재사용 선택 MAPE (%)'],direct),'',
        '직접 학습의 네 조건 최저값은 모두 MLP32·k=30이다. 재사용 구성은 같은 개발 자료에서 더 낮은 오차를 냈다. source의 2,652형상 사전 비용은 이미 지불한 공통 준비비이므로 새로운 자료만으로 학습한 D0와 전체 구축비가 같다고 설명하지 않는다.','',
        '잔차 표현 자체를 확인하는 대조군은 D+A다. 예를 들어 RF FFN FP16·k=20에서 D0/D+A/R의 MAPE는 41.070/36.546/5.531%다. 두 prior 사용 모델의 입력은 같지만 목표 표현이 다르다. RF 직접 모델의 형상 외삽 한계와 작은 레이어 학습량에서의 차이를 포함한 관측이며 모든 모델에 대한 보편적 정리를 뜻하지 않는다.','',
        f"![RQ3 residual learning curves]({OUT/'residual_learning_curve.png'})",'',
        f"![RQ3 direct learning curves]({OUT/'direct_learning_curve.png'})",'',
        '### 검증 형상 수에 따른 불확실성','',
        table(['대상','선택 MAPE','검증 그룹 bootstrap 95% 구간','세 seed MAPE 범위'],[
            [r['layer']+' '+r['dtype'],f"{r['mape_pct']:.2f}%",f"{r['group_bootstrap_ci95_low']:.2f}~{r['group_bootstrap_ci95_high']:.2f}%",f"{r['seed_mape_min']:.2f}~{r['seed_mape_max']:.2f}%"] for r in uncertainty]),'',
        '동일한 10검증 그룹의 seed 평균 오차를 그룹 단위로 10,000회 재표집했다. 30개 독립 표본으로 간주하지 않았다. 선택한 모델을 고정한 개발 자료의 민감도이며 모델 선택 편향을 보정한 구간이나 새 형상 일반화 성능의 보장이 아니다.','',
        '## RQ2 구현 방식 차이','',
        table(['대상','조건 수','분해/표준 중앙값','비율 범위','±5% 이내'],[
            [r['layer']+' '+r['dtype'],r['conditions'],f"{r['median_ratio']:.4f}",f"{r['min_ratio']:.3f}~{r['max_ratio']:.3f}",r['within_5pct_count']] for r in gaps]),'',
        '비율이 1보다 작으면 이번 명시적 분해 구현이 더 빠르다. 중앙값 차이는 대체로 작지만 형상별 차이는 더 컸다. 특히 Linear FP16은 0.658~1.296으로 경로 간 시간을 항상 같게 취급할 수 없다. 이 비율은 구현 방식의 차이이며 예측 오차나 순수 조립 오차가 아니다. Conv의 새 확인·구성요소 실험은 이번 ZIP에 없다.','',
        f"![RQ2 implementation ratios]({RESULT/'implementation_ratios.png'})",'',
        '## 비용과 결과의 범위','',
        table(['공통 k','한 seed 표준 원측정','세 seed train 합집합 그룹','세 seed 합집합 표준 원측정'],[
            [r['k'],r['per_seed_standard_records'],r['three_seed_train_union_groups'],r['three_seed_union_standard_records']] for r in costs]),'',
        f"표준 원측정은 세 회차를 포함한다. 모든 계열이 같은 정답을 공유하므로 학습기 수나 구성 수만큼 GPU 비용을 곱하지 않는다. 네 조건별 정확도 우선 k를 따로 적용하면 한 seed의 고유 조건 원측정은 {cost['selected_per_seed_distinct_condition_records']}개, 세 seed의 합집합은 {cost['selected_three_seed_union_distinct_condition_records']}개다. 그러나 현재 묶음 수집기는 각 그룹의 두 레이어·두 dtype를 함께 측정하고 FFN FP32가 train 30그룹을 모두 필요로 하므로 표준 480측정을 수집했다. explicit 짝까지 합친 실제 비용은 960측정·66.20분이다. 330개 등의 회고적 레이블 예산을 실제 GPU 시간 절감이라고 부르지 않는다.",'',
        f"관측된 timing setup와 본측정 host block 합은 표준 {cost['observed_standard_timing_setup_and_host_blocks_seconds']:.2f}초, explicit {cost['observed_explicit_timing_setup_and_host_blocks_seconds']:.2f}초다. pair 생성과 사전 정확성의 알려진 공통 부분은 중복 없이 {cost['observed_shared_pair_creation_and_before_correctness_seconds']:.2f}초이며 사후 정확성·180초 preheat·감시와 기타 overhead는 이 부분합에 포함하지 않았다. 기존 GEMM source 수집의 정확한 GPU wall이 없으므로 전체 사전 비용의 절감·손익분기점을 계산하지 않았다. 네 학습기 레이어 비교의 로컬 CPU wall은 {c.layer.read(OUT/'selection_frozen.json')['wall_seconds']:.2f}초, 실제 model fit 합은 {c.layer.read(OUT/'selection_frozen.json')['total_model_fit_seconds']:.2f}초다.",'',
        '이번 결과는 새 레이어 정답에서도 GEMM prior의 재사용과 보정이 유용할 수 있다는 개발 근거다. 기존 RQ1/RQ4의 큰 GEMM 검증과 새 레이어 10그룹 검증을 같은 모집단으로 섞지 않는다. 선행연구 대비 정확도 우월성, 미열람 시험의 최종 성능, 실제 전체 수집비 절감은 아직 입증하지 않았다. 추가 GEMM 10,000형상 수집 결과가 도착하면 기존 네 학습기 protocol로 보강 효과를 별도 평가한다.','',
        '## 파일과 재현','',
        f"- [전체 원자료 검산]({RESULT/'verification_all.json'}) / [검산 코드]({RESULT/'verify_result.py'})",
        f"- [네 학습기 고정 protocol]({OUT/'frozen_protocol.json'}) / [선택 동결 시각]({OUT/'selection_frozen.json'})",
        f"- [조건별 집계]({OUT/'aggregate_metrics.csv'}) / [9,600개 validation 예측]({OUT/'validation_predictions.csv'})",
        f"- [원 RF 예측·87개 가중치 독립 확인]({OUT/'independent_verification.json'}) / [비용]({OUT/'cost_summary.json'})",
        f"- [형상별 선택 모델 오차]({OUT/'selected_group_errors.csv'}) / [선택 모델]({OUT/'models'})",'',
        '원본 source와 이전 full 문서, 기존 학습·선택과 잠금은 보존했다. 입력 해시는 실행 전후 일치하며 네 학습기의 무효 예측·수렴 경고는 모두 0개다.']
    (c.layer.HERE/'레이어_개발실험_결과_2026-10-28.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
