"""Summarize the frozen comparison; no original timing source is read."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import compare_learners as c

OUT=c.HERE/'runs/v1'


def read(name):
    with (OUT/name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    completed=json.loads((OUT/'execution_completed.json').read_text())
    rows=read('aggregate_metrics.csv');fits=read('fit_diagnostics.csv');scores=read('metrics.csv')
    summaries=[]
    for n in [2652,3345]:
        for learner in c.LEARNERS:
            for target in c.TARGETS:
                for dtype in c.old.DTYPES:
                    selected=[r for r in rows if r['rq']=='RQ1' and r['learner']==learner and r['method']==target and int(r['n'])==n and r['dtype']==dtype]
                    assert len(selected)==3
                    summaries.append(dict(n=n,learner=learner,target=target,dtype=dtype,
                        mean_mode_mape_pct=np.mean([float(r['mean_mape_pct']) for r in selected]),
                        mean_mode_p90_pct=np.mean([float(r['mean_p90_pct']) for r in selected])))
    c.old.save_csv(OUT/'endpoint_summary.csv',summaries)
    accuracy=[]
    for rq in ['RQ1','RQ4']:
        modes=c.old.MODES if rq=='RQ1' else ['prealloc','graph']
        for dtype in c.old.DTYPES:
            for mode in modes:
                group=[r for r in rows if r['rq']==rq and r['dtype']==dtype and r['mode']==mode and int(r['failures'])==0]
                chosen=min(group,key=lambda r:(float(r['mean_mape_pct']),float(r['mean_p90_pct']),int(r['n']),r['learner'],r['method']))
                accuracy.append({**chosen,'effective_target_calibration_shapes':0 if chosen['method']=='zero' else chosen['n'] if rq=='RQ4' else '',
                                 'selection_policy':'User preference: minimum development MAPE, then p90; no 15/30 stopping gate'})
    c.old.save_csv(OUT/'accuracy_first_selection_validation_only.csv',accuracy)
    text=['# MLP 두 크기, RF와 Ridge의 동일 자료 비교','',
          '2026-10-28 사용자 요청에 따라 네 학습기를 같은 형상, 특징, 정답과 검증 구역에서 비교했다. 원래 RF 실행본과 가중치는 보존했다. 이번 결과는 개발 validation이며 잠금 시험은 열지 않았다.','',
          'RQ1은 학습기 4개 × 목표 2개 × 자료량 5개 × seed 3개 × 정밀도 2개 × mode 3개의 720구성이다. RQ4는 source 학습기 4개 × 방법 5개 × 보정 예산 5개 × seed 3개 × 정밀도 2개 × target mode 2개의 1,200구성이다. 전체 1,920개 구성을 평가했다. 실제 회귀기 fit은 1,440회이고 나머지 480구성은 무보정 또는 상수 보정이다.','',
          '## 학습 조건','',
          table(['학습기','고정 설정'],[
              ['RF','100 trees, max_depth=8, min_samples_leaf=2, n_jobs=1'],
              ['Ridge','alpha=1, 학습 부분집합으로만 입력 표준화'],
              ['MLP32','15→32→32→1, ReLU, 1,601 parameters'],
              ['MLP64','15→64→64→1, ReLU, 5,249 parameters']]),'',
          '두 MLP는 Adam, learning_rate=0.001, alpha=0.0001, batch=min(128,n), 최대 3,000 epoch, 학습 손실 tol=1e-6와 patience=100을 사용했다. 입력과 log 정답의 표준화도 현재 학습 부분집합 안에서만 fit했다. 외부 validation을 epoch 선택이나 조기 종료에 사용하지 않았다. 모든 학습 표본을 fit에 사용했고 각 모델의 수렴 경고와 실제 epoch를 저장했다. MLP 설정의 epoch·수렴 의미는 [scikit-learn 1.8 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPRegressor.html)를 따른다.','',
          '공통 입력은 기존 15개 정적 특징이며 Roofline 상수와 단위도 이전 실행과 같다. 직접 목표는 log(T), 정규화 목표는 log(T/L)이고 예측 시 각각 exp(output), L·exp(output)으로 복원했다. RF와 Ridge는 이전 설정을 유지했고 추가 튜닝은 하지 않았다. 이 비교는 정해 둔 설정끼리의 비교이며 각 모델 계열의 최적 성능 또는 같은 계산시간 예산을 보장하는 비교가 아니다.','',
          '사용자는 후속 목표 질문에 “목표 수치보다 가능한 만큼 낮추기”를 선택했다. 이에 기존 15%/30% 통과 여부와 최소 비용 선택은 원래 protocol의 참고 결과로 남기고, 현재 추천은 같은 개발 validation에서 MAPE가 가장 낮은 구성, 동률이면 p90이 낮은 구성으로 정리한다. 이 정책은 사용자의 후속 선호를 반영한 파생 선택이며 최초 protocol을 소급 수정한 것이 아니다. 정해진 비교 격자 안의 최저값이고 최적화 가능한 전역 최저 오차를 입증하지 않는다.','',
          '## RQ1 자료량별 결과','',
          'A 2,652형상은 기존에 C를 제외한 학습 구역 전체다. 3,345형상은 기존 legacy_train에서 두 dtype과 세 mode가 완비된 모든 형상이며 RQ1에만 추가했다. 작은 부분집합은 이전 seed별 A 순서를 그대로 사용하고 3,345 endpoint에는 C 693개를 뒤에 붙였다. 두 endpoint를 같은 자료량으로 혼동하지 않는다. validation은 모두 기존 716형상이다.','']
    for n in [2652,3345]:
        values=[]
        for learner in c.LEARNERS:
            for target in c.TARGETS:
                chosen=[next(r for r in summaries if r['n']==n and r['learner']==learner and r['target']==target and r['dtype']==d) for d in c.old.DTYPES]
                values.append([learner,target]+[f"{r['mean_mode_mape_pct']:.3f} / {r['mean_mode_p90_pct']:.3f}" for r in chosen])
        text += [f'### n={n}', '', table(['학습기','목표','FP16 MAPE / p90 (%)','FP32 MAPE / p90 (%)'],values),'',
                 '각 값은 세 seed를 평균한 뒤 세 mode를 동일 가중 평균한 것이다. 개별 mode의 p90은 조건부 오차 분위수이며 표의 평균 p90을 전체 표본을 합친 p90으로 해석하지 않는다.','']
    text += ['### 학습곡선','',
             '가로축은 고유 학습 형상 수이고 세로축은 상대오차다. 두 축에 log scale을 적용했으며 실선은 직접 목표, 점선은 정규화 목표다. 각 점은 세 seed 평균이며 다른 정밀도와 mode의 조건을 한 모델로 합치지 않았다.','',
             f"![MAPE learning curves]({OUT/'learning_curve_mape_pct.png'})",'',
             f"![p90 learning curves]({OUT/'learning_curve_p90_pct.png'})",'']
    endpoint=[]
    for r in rows:
        if r['rq']=='RQ1' and r['n']=='3345':
            endpoint.append([r['learner'],r['method'],r['dtype'],r['mode'],f"{float(r['mean_mape_pct']):.3f}",f"{float(r['mean_p90_pct']):.3f}"])
    text += ['### 3,345형상의 개별 실행 조건','',table(['학습기','목표','dtype','mode','MAPE (%)','p90 (%)'],endpoint),'',
             '## RQ4 비교','',
             '각 학습기와 dtype별로 A 2,652형상에서 학습한 직접/정규화 source를 allocate validation의 세 seed 평균 MAPE로 선택했다. 이 선택은 target label 접근 전에 파일로 저장했다. Source는 고정하고 calibration에서는 같은 학습기 계열의 log 잔차, target 직접 예측과 source 예측을 추가 입력으로 받는 직접 예측을 비교했다. 상수 비율 보정과 무보정도 포함했다. MLP source 가중치를 미세조정한 실험은 아니다.','',
             'C의 seed별 동일 중첩 5/10/30/100/300형상과 같은 target validation 30형상을 사용했다. 목표 기준은 세 seed 평균 MAPE≤15%, 평균 p90≤30%이며 예측 실패가 없어야 한다. 실제 보정 자료가 가장 적은 구성을 선택하고 무보정의 비용은 0으로 처리했다. 모든 target validation label은 이전 개발 작업에서도 접근한 이력이 있으므로 이번 수치를 새 미열람 일반화 결과로 주장하지 않는다.','']
    text += ['RQ4 단독의 세 seed 보정 표본 합집합은 558형상이며 target validation은 30형상이다. 네 학습기가 같은 label을 공유하므로 GPU label 비용을 학습기 수만큼 곱하지 않는다. 이후 RQ1의 3,345형상 endpoint를 계산하면서 C 693형상의 전체 mode 정답을 읽었다. 따라서 이번 연구 전체가 C 558형상만 소비했다고 해석하면 안 된다. 학습 전체는 20,070조건, 기존 validation은 4,296조건이며 합계 24,366개의 고유 조건 정답을 사용했다. 이는 세 회차를 집계한 값이고 대응하는 원측정은 73,098건이다.','']
    selected=read('rq4_selections_validation_only.csv')
    text += [table(['학습기','dtype','target','방법','보정 형상','검증 형상','MAPE (%)','p90 (%)','목표 도달'],
                   [[r['learner'],r['dtype'],r['mode'],r['method'],r['effective_calibration_shapes'],r['validation_shapes'],f"{float(r['mean_mape_pct']):.3f}",f"{float(r['mean_p90_pct']):.3f}",r['goal_attained']] for r in selected]),'',
             '## 오차 최소화 우선의 현재 선택','',
             '아래 표는 고정된 비교 격자 전체에서 mean MAPE를 최소화하는 구성이다. RQ4의 n은 source 학습 2,652개에 추가하는 target 보정 표본이고 validation 30개도 별도 비용이다. RQ1의 3,345 source를 RQ4에 섞지 않았다. 15%/30%를 이미 통과했다는 이유로 더 낮은 오차의 구성을 제외하지 않는다.','',
             table(['RQ','dtype','mode','학습기','방법','n/k','MAPE (%)','p90 (%)'],
                   [[r['rq'],r['dtype'],r['mode'],r['learner'],r['method'],r['n'],f"{float(r['mean_mape_pct']):.3f}",f"{float(r['mean_p90_pct']):.3f}"] for r in accuracy]),'',
             '## 학습 오차, 수렴과 비용','']
    diagnostic=[]
    for learner in c.LEARNERS:
        sel=[r for r in fits if r['rq']=='RQ1' and r['learner']==learner and r['n']=='3345']
        diagnostic.append([learner,f"{np.mean([float(r['train_mape_pct']) for r in sel]):.3f}",
            f"{np.mean([float(r['mape_pct']) for r in scores if r['rq']=='RQ1' and r['learner']==learner and r['n']=='3345']):.3f}",
            f"{np.median([float(r['fit_seconds']) for r in sel]):.3f}",
            f"{min(int(r['epochs']) for r in sel)}–{max(int(r['epochs']) for r in sel)}" if learner.startswith('mlp') else '해당 없음',
            sum(int(r['convergence_warnings'])>0 for r in sel)])
    text += [table(['학습기','평균 train MAPE (%)','평균 validation MAPE (%)','fit 중앙값 (s)','epoch 범위','수렴 경고 fit'],diagnostic),'',
             '위 진단 표는 직접/정규화, 두 dtype, 세 mode와 세 seed를 모두 평균한 요약이며 특정 한 설정의 성능이 아니다. 훈련 오차와 검증 오차의 차이는 형상 수만 늘리면 해결된다는 증명이 아니다. 큰 MLP의 훈련 오차가 낮아도 검증 오차가 낮아지지 않으면 더 많은 형상, 특징과 정규화의 후속 가설을 구분해 확인해야 한다.','',
             f"전체 실행 wall time은 {completed['wall_seconds']:.2f}초, fit 시간 합은 {completed['total_fit_seconds']:.2f}초다. 수렴 경고가 발생한 구성은 {completed['convergence_warning_fits']}개, 유효하지 않은 validation 예측은 {completed['invalid_predictions']}개다. 경고를 숨기거나 해당 구성을 결과에서 삭제하지 않았다. 이 비용은 로컬 CPU fit 비용이며 GPU 측정 수집시간이 아니다.",'',
             '### 최대 epoch 도달 구성의 별도 수렴 확인','']
    audit=c.HERE/'runs/convergence_audit_v1'
    if (audit/'completed.json').exists():
        done=json.loads((audit/'completed.json').read_text())
        with (audit/'metrics.csv').open(newline='') as handle:
            extra=list(csv.DictReader(handle))
        text += [f"주 비교에서 수렴 경고가 발생한 {done['audited']}개 MLP 구성만 같은 seed와 자료로 처음부터 다시 학습하되 최대 epoch를 10,000으로 늘렸다. 검증 오차와 관계없이 모든 경고 구성을 대상으로 했고 학습 손실 종료 규칙은 그대로다. 추가 수렴 확인 뒤 경고가 남은 구성은 {done['remaining_warning_fits']}개다. 이 추가 예산의 결과는 별도 audit이며 주 비교의 고정 3,000 epoch 결과를 덮어쓰지 않았다.",'',
                 table(['RQ','학습기','dtype','mode','방법','n','seed','기존 MAPE (%)','추가 MAPE (%)','추가 epoch','남은 경고'],
                       [[r['rq'],r['learner'],r['dtype'],r['mode'],r['method'],r['n'],r['seed'],f"{float(r['primary_mape_pct']):.3f}",f"{float(r['mape_pct']):.3f}",r['epochs'],r['convergence_warnings']] for r in extra]),'']
    else:
        text += ['수렴 경고 구성의 추가 확인은 아직 완료되지 않았다. 주 비교 표를 완전 수렴한 최적해의 비교로 해석하지 않는다.','']
    text += [
             '## 추가 10,000형상','',
             '사용자는 새 형상 10,000개 추가, 두 dtype·세 mode·세 회차의 180,000측정을 선택했다. [별도 수집 실행본](../../gemm_expand_20261028_v1/README.md)에 9,000 train / 1,000 새 development validation을 준비했다. 기존 6,002 main 형상, 진단 형상과 보호된 레이어 형상을 시간값 없이 제외했다. 기존 학습 3,345개와 새 학습 9,000개가 모두 품질 검사를 통과하면 최대 학습 풀은 12,345형상이다. 기존 시험과 새 검증을 훈련에 합치지 않는다.','',
             '코드·목록·명령 준비와 CPU 정적 검사를 수행했으며 새 GPU 측정은 아직 없다. 보강 자료 반환 뒤 같은 고정 학습기 설정으로 기존 validation과 새 validation을 각각 평가한다. RQ4의 기존 source A/C 분리는 이번 full RQ1 endpoint와 독립적으로 유지한다.','',
             '## 보존과 해석 범위','',
             '기존 raw 자료, PASS/REVIEW 이력, 이전 full 컨텍스트와 RF 실행본을 덮어쓰지 않았다. 최종 GEMM 시험과 레이어 잠금 시험은 열지 않았다. 현재 표는 고정 설정의 개발 성능이며 실제 레이어 일반화, 새 GPU/클럭과 커널 ID·tile 구조를 이용한 개선을 입증하지 않는다. 모델 선택을 새로 비교했으므로 기존 소량 보정 자료 절약 주장을 모든 학습기에 자동 확대하지 않는다.','',
             '원시 검증 예측은 `runs/v1/validation_predictions.csv`, 학습 및 수렴은 `fit_diagnostics.csv`, 층별 오류는 `subgroup_metrics.csv`, 정규화의 paired bootstrap은 `paired_target_bootstrap.csv`다. 두 최대 자료량의 모델·전처리 가중치와 protocol 참조는 `models/`에 저장했고 저장 직후 예측의 직렬화 전후 일치를 확인했다.','']
    (c.HERE/'동일조건_학습기비교_결과_2026-10-28.md').write_text('\n'.join(text))
    colors=dict(rf='#2364AA',ridge='#999999',mlp32='#E67E22',mlp64='#2A9D8F')
    for metric,goal in [('mape_pct',15),('p90_pct',30)]:
        fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True)
        for i,dtype in enumerate(c.old.DTYPES):
            for j,mode in enumerate(c.old.MODES):
                ax=axes[i,j]
                for learner in c.LEARNERS:
                    for target in c.TARGETS:
                        chosen=sorted([r for r in rows if r['rq']=='RQ1' and r['dtype']==dtype and r['mode']==mode and r['learner']==learner and r['method']==target],key=lambda r:int(r['n']))
                        ax.plot([int(r['n']) for r in chosen],[float(r['mean_'+metric]) for r in chosen],
                            color=colors[learner],ls='-' if target=='direct' else '--',marker='o',ms=3,label=f'{learner} {target}')
                ax.axhline(goal,color='#777777',ls=':',lw=.8)
                ax.set_xscale('log');ax.set_title(f'{dtype.upper()} / {mode}')
                ax.set_yscale('log')
                ax.set_ylabel('MAPE (%)' if metric=='mape_pct' else 'p90 relative error (%)')
                ax.grid(alpha=.18)
                if i==1:ax.set_xlabel('Unique training shapes')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=8)
        fig.tight_layout(rect=(0,0,1,.91))
        fig.savefig(OUT/f'learning_curve_{metric}.png',dpi=170)
        fig.savefig(OUT/f'learning_curve_{metric}.pdf')
        plt.close(fig)
    print('PASS: result report, endpoint tables and learning-curve figures generated from saved results')


if __name__=='__main__':
    main()
