"""Audit saved comparisons and create reproducible tables and figures."""
from collections import defaultdict
import csv
import itertools
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import adapters as a
from compare import read, csv_save
from prepare import HERE, ROOT, REFERENCE, save, sha

LAYER = HERE/'runs/v1'
GEMM = HERE/'gemm/v1'
TABLES = HERE/'tables'
FIGURES = HERE/'figures'
COMMON = ['linear', 'conv1', 'conv3']
MAIN = ['neuralpower', 'gemmify_dt', 'gemmify_capacity', 'paleo_empirical', 'neusight_clock']
NAMES = {'ours_overall': 'Ours: frozen overall', 'ours_direct': 'Ours: local direct',
    'neuralpower': 'NeuralPower port', 'neuralpower_fc_port': 'NeuralPower FC port',
    'gemmify_dt': 'GEMMify: literal grid', 'gemmify_capacity': 'GEMMify: capacity variant',
    'paleo_empirical': 'Paleo: calibrated', 'neusight_clock': 'NeuSight: pretrained',
    'neusight_bmm_clock': 'NeuSight: BMM proxy', 'neusight_linear_clock': 'NeuSight: LINEAR proxy'}
COLORS = dict(zip(['ours_overall','ours_selected','ours_direct','neuralpower','neuralpower_fc_port',
    'gemmify_dt','gemmify_capacity','paleo_empirical','neusight_bmm_clock','neusight_linear_clock'],
    ['#1f77b4','#1f77b4','#ff7f0e','#2ca02c','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']))


def csv_read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def group(rows, fields):
    result = defaultdict(list)
    for r in rows:
        result[tuple(r[f] for f in fields)].append(r)
    return result


def audit():
    count = 0
    for name, digest in read(HERE/'original_inputs.json').items():
        assert sha(ROOT/name) == digest, name
    for name, digest in read(HERE/'source_hashes.json').items():
        assert sha(HERE/name) == digest, name
    for folder in [LAYER, GEMM]:
        frozen = read(folder/'selection_frozen.json')
        for name, digest in frozen['artifacts'].items():
            assert sha(folder/name) == digest, name
        for name, digest in frozen['code'].items():
            assert sha(HERE/name) == digest, name
        fields = ['target', 'layer', 'dtype', 'method', 'k', 'seed'] if folder == LAYER else ['dtype', 'mode', 'method', 'seed']
        identity = 'group_id' if folder == LAYER else 'shape'
        for stage, prefix in [(folder, 'validation'), (folder/'evaluation', 'test')]:
            predictions = read(stage/(prefix+'_predictions.json'))
            groups = group(predictions, fields)
            metrics = read(stage/(prefix+'_metrics.json'))
            assert len(groups) == len(metrics)
            for metric in metrics:
                rows = groups[tuple(metric[f] for f in fields)]
                assert len({r[identity] for r in rows}) == len(rows) == metric['groups']
                actual = np.array([float(r['actual_ms']) for r in rows])
                predicted = np.array([float(r['predicted_ms']) for r in rows])
                assert np.all(actual > 0) and np.all(np.isfinite(predicted)) and np.all(predicted > 0)
                error = 100*np.abs(predicted-actual)/actual
                expected = [np.mean(error), np.percentile(error, 90), np.sqrt(np.mean(error**2))]
                np.testing.assert_allclose(expected, [metric[k] for k in ['mape_pct', 'p90_pct', 'rmspe_pct']], rtol=1e-12, atol=1e-10)
                count += len(rows)
    dev = read(HERE/'development.json')
    train = {tuple([r['case']['group_id'], r['case']['layer'], r['dtype'], r['target']]) for r in dev if r['case']['split']=='train'}
    valid = {tuple([r['case']['group_id'], r['case']['layer'], r['dtype'], r['target']]) for r in dev if r['case']['split']=='validation'}
    assert not train & valid
    refs = read(LAYER/'model_references.json')
    validation = read(LAYER/'validation_predictions.json')
    by_ref = group([r for r in validation if r['method'] in ['neuralpower', 'gemmify_dt', 'gemmify_capacity']], ['model_ref'])
    case_map = {(r['case']['group_id'], r['case']['layer']): r['case'] for r in dev}
    reloaded = 0
    for ref in refs:
        assert len(ref['train_keys']) == ref['train_labels']
        assert set(map(tuple, ref['train_keys'])) <= train
        assert not set(map(tuple, ref['train_keys'])) & valid
        rows = by_ref[ref['model_ref'],]
        model = joblib.load(LAYER/ref['model_ref'])
        got = a.predict_fit(model, [case_map[r['group_id'], r['layer']] for r in rows])
        np.testing.assert_allclose(got, [r['predicted_ms'] for r in rows], rtol=1e-12, atol=1e-12)
        reloaded += len(rows)
    choices = group(read(LAYER/'validation_aggregate.json'), ['target','layer','dtype','method'])
    for chosen in read(LAYER/'selected.json'):
        candidates = [r for r in choices[tuple(chosen[f] for f in ['target','layer','dtype','method'])] if not r['failures']]
        assert chosen == min(candidates, key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'],r['mean_train_labels'],r['k']))
    test = {(r['group_id'],r['layer'],r['dtype'],r['target']) for r in read(LAYER/'evaluation/test_predictions.json')}
    assert not test & (train | valid)
    from gemm_compare import cases
    gemm_rows = group([r for r in read(GEMM/'validation_predictions.json') if r['method']=='neuralpower_fc_port'], ['dtype','mode','seed'])
    for ref in read(GEMM/'models.json'):
        rows = gemm_rows[ref['dtype'],ref['mode'],ref['seed']]
        shapes = [tuple(map(int,r['shape'].split('x'))) for r in rows]
        got = a.predict_fit(joblib.load(GEMM/ref['model_ref']), cases(shapes,'validation'))
        np.testing.assert_allclose(got, [r['predicted_ms'] for r in rows], rtol=1e-12, atol=1e-12)
        reloaded += len(rows)
    p = read(ROOT/'analysis/gemm_expansion_result_2026-11-06/fit_v1/frozen_protocol.json')
    train_shapes = {tuple(s) for s in p['old_train_orders'][str(p['seeds'][0])]+p['new_train_orders'][str(p['seeds'][0])]}
    val_shapes = {tuple(s) for s in p['legacy_validation_shapes']+p['new_validation_shapes']}
    test_shapes = {tuple(map(int,r['shape'].split('x'))) for r in read(GEMM/'evaluation/test_predictions.json')}
    assert not train_shapes & val_shapes and not (train_shapes | val_shapes) & test_shapes
    result = dict(status='PASS',original_inputs_unchanged=len(read(HERE/'original_inputs.json')),
        source_files_unchanged=len(read(HERE/'source_hashes.json')),metric_predictions_recomputed=count,
        saved_models_reloaded=len(refs)+len(read(GEMM/'models.json')),saved_validation_predictions_reproduced=reloaded,
        layer_choices_verified=len(read(LAYER/'selected.json')),training_validation_test_overlap=0,
        bootstrap='5000 paired shape resamples; errors averaged over fixed seeds first; descriptive 95% percentile interval without multiplicity correction')
    save(HERE/'verification.json', result)
    return result


def paired(reference, baseline, identity, info):
    left, right = group(reference,[identity]), group(baseline,[identity])
    assert left.keys() == right.keys()
    delta = []
    for key in sorted(left):
        ys = [float(r['actual_ms']) for r in left[key]+right[key]]
        assert len(set(ys)) == 1
        error = lambda rows: np.mean([abs(float(r['predicted_ms'])/float(r['actual_ms'])-1)*100 for r in rows])
        delta.append(error(right[key])-error(left[key]))
    delta = np.array(delta)
    rng = np.random.default_rng(20261111)
    means = np.mean(delta[rng.integers(0,len(delta),size=(5000,len(delta)))],axis=1)
    low,high = np.percentile(means,[2.5,97.5])
    return dict(info,groups=len(delta),baseline_minus_reference_mape_pp=float(delta.mean()),
                ci95_low_pp=float(low),ci95_high_pp=float(high),resamples=5000)


def layer_tables():
    own = csv_read(ROOT/'analysis/final_test_2026-11-11/summary/layers.csv')
    own_pred = csv_read(ROOT/'analysis/final_test_2026-11-11/layers/test_predictions.csv')
    new = read(LAYER/'evaluation/test_aggregate.json')
    new_pred = read(LAYER/'evaluation/test_predictions.json')
    choices = read(REFERENCE/'prepared_layer_full/selected_configurations.json')
    rows = []
    for r in own:
        if r['selection_scope'] not in ['overall','direct']: continue
        choice = next(x for x in choices if all(x[k]==r[k] for k in ['target','layer','dtype','selection_scope']))
        rows.append(dict(target=r['target'],layer=r['layer'],dtype=r['dtype'],method='ours_'+r['selection_scope'],
            k=int(r['k']),train_labels_per_model=choice['actual_train_labels'],seeds=int(r['seeds']),groups=int(r['test_groups']),
            mape_pct=float(r['mean_mape_pct']),p90_pct=float(r['mean_p90_pct']),validation_mape_pct=float(r['validation_mape_pct']),
            recipe=choice['learner']+'/'+choice['method'],local_source_cost='shared GEMM source; separate ledger' if choice['method'] in ['zero','residual','prior_direct','constant_ratio'] else 'none'))
    selected = {(r['target'],r['layer'],r['dtype'],r['method']):r for r in read(LAYER/'selected.json')}
    for r in new:
        if not r['selected_on_validation']: continue
        key = tuple(r[k] for k in ['target','layer','dtype','method'])
        rows.append(dict(target=r['target'],layer=r['layer'],dtype=r['dtype'],method=r['method'],k=r['k'],
            train_labels_per_model=r['mean_train_labels'],seeds=r['seeds'],groups=49 if r['layer']=='conv3' and r['dtype']=='fp32' else 50,
            mape_pct=r['mean_mape_pct'],p90_pct=r['mean_p90_pct'],validation_mape_pct=selected[key]['mean_mape_pct'],
            recipe=r['method'],local_source_cost='external/unknown' if r['method'].startswith('neusight') else 'existing hardware calibration' if r['method'].startswith('paleo') else 'none'))
    csv_save(TABLES/'layer_selected_test.csv',rows)
    csv_save(TABLES/'layer_k200_test.csv',[r for r in new if r['k']==200])
    comparisons = []
    for r in new:
        if not r['selected_on_validation'] or r['method'] not in MAIN: continue
        baseline=[x for x in new_pred if all(x[k]==r[k] for k in ['target','layer','dtype','method','k'])]
        for scope in ['overall','direct']:
            reference=[x for x in own_pred if x['selection_scope']==scope and all(x[k]==r[k] for k in ['target','layer','dtype'])]
            comparisons.append(paired(reference,baseline,'group_id',dict(target=r['target'],layer=r['layer'],dtype=r['dtype'],baseline=r['method'],reference='ours_'+scope)))
    csv_save(TABLES/'layer_paired_bootstrap.csv',comparisons)
    return rows,comparisons


def gemm_tables():
    own = [r for r in csv_read(ROOT/'analysis/final_test_2026-11-11/gemm_expanded/predictions.csv') if r['variant']=='selected']
    rows = []
    for (dtype,mode),pred in group(own,['dtype','mode']).items():
        seed_scores=[]
        for _,rs in group(pred,['seed']).items():
            e=np.array([abs(float(r['predicted_ms'])/float(r['actual_ms'])-1)*100 for r in rs])
            seed_scores.append((e.mean(),np.percentile(e,90)))
        rows.append(dict(dtype=dtype,mode=mode,method='ours_selected',seeds=len(seed_scores),train_labels=11498,groups=1000,
                         mean_mape_pct=float(np.mean([x[0] for x in seed_scores])),mean_p90_pct=float(np.mean([x[1] for x in seed_scores]))))
    new=read(GEMM/'evaluation/test_aggregate.json')
    rows.extend(new)
    csv_save(TABLES/'gemm_test.csv',rows)
    comparisons=[]
    new_pred=read(GEMM/'evaluation/test_predictions.json')
    for r in new:
        if r['method'] not in ['neuralpower_fc_port','paleo_empirical','neusight_bmm_clock','neusight_linear_clock']:continue
        reference=[x for x in own if x['dtype']==r['dtype'] and x['mode']==r['mode']]
        baseline=[x for x in new_pred if all(x[k]==r[k] for k in ['dtype','mode','method'])]
        comparisons.append(paired(reference,baseline,'shape',dict(dtype=r['dtype'],mode=r['mode'],baseline=r['method'],reference='ours_selected')))
    csv_save(TABLES/'gemm_paired_bootstrap.csv',comparisons)
    return rows,comparisons


def budgets():
    refs=read(LAYER/'model_references.json')
    dev=read(HERE/'development.json')
    choices=read(REFERENCE/'prepared_layer_full/selected_configurations.json')
    direct={(r['target'],r['layer'],r['dtype']):r for r in choices if r['selection_scope']=='direct'}
    candidates=read(LAYER/'validation_aggregate.json')
    def label_set(row):
        method='neuralpower' if row['method']=='neuralpower_original_floor' else row['method']
        family='conv' if row['layer'].startswith('conv') else 'fc'
        return {tuple(key) for r in refs if r['method']==method and r['k']==row['k']
            and r['dtype']==row['dtype'] and r['target']==row['target'] and r['family'] in ['all',family] for key in r['train_keys']}
    recipes=[]
    for r in read(LAYER/'selected.json'):
        v=sum(x['target']==r['target'] and x['dtype']==r['dtype'] and x['case']['layer']==r['layer'] and x['case']['split']=='validation' for x in dev)
        labels=label_set(r)
        recipes.append(dict(target=r['target'],layer=r['layer'],dtype=r['dtype'],method=r['method'],k=r['k'],
            train_labels_per_model=r['mean_train_labels'],unique_training_labels_three_seeds=len(labels),validation_labels=v,
            recipe_development_labels_union=len(labels)+v,raw_measurements_three_rounds=3*(len(labels)+v)))
    csv_save(TABLES/'layer_selected_label_recipes.csv',recipes)
    rows=[]
    for key,rs in group(candidates,['target','layer','dtype','method']).items():
        t,l,d,m=key; reference=direct[t,l,d]
        goals={'direct_mape':(reference['mean_mape_pct'],float('inf')),
               'direct_mape_and_p90':(reference['mean_mape_pct'],reference['mean_p90_pct']),
               'mape_10':(10,float('inf')),'mape_15':(15,float('inf')),'mape_20':(20,float('inf')),'legacy_15_30':(15,30)}
        for goal,(mape,p90) in goals.items():
            ok=[r for r in rs if not r['failures'] and r['mean_mape_pct']<=mape+1e-12 and r['mean_p90_pct']<=p90+1e-12]
            base=dict(target=t,layer=l,dtype=d,method=m,goal=goal,target_mape_pct=mape,target_p90_pct=p90 if np.isfinite(p90) else None,reached=bool(ok))
            if ok:
                best=min(ok,key=lambda r:(len(label_set(r)),r['mean_mape_pct'],r['mean_p90_pct'],r['k']))
                base.update(k=best['k'],train_labels_per_model=best['mean_train_labels'],unique_training_labels_three_seeds=len(label_set(best)),validation_mape_pct=best['mean_mape_pct'],validation_p90_pct=best['mean_p90_pct'])
            rows.append(base)
    csv_save(TABLES/'layer_equivalent_accuracy.csv',rows)
    costs=[]
    for scope,folder,filename in [('layers',LAYER,'fit_diagnostics.json'),('gemm',GEMM,'diagnostics.json')]:
        ds=read(folder/filename); frozen=read(folder/'selection_frozen.json')
        costs.append(dict(scope=scope,fit_calls=frozen['fit_calls'],wall_seconds=frozen['wall_seconds'],
            fit_call_seconds=sum(x.get('fit_seconds',0) for x in ds),logged_prediction_seconds=sum(x.get('prediction_seconds',0) for x in ds),
            convergence_warnings=sum(x.get('convergence_warnings',0) for x in ds)))
    csv_save(TABLES/'cpu_observed_calls.csv',costs)
    return rows,costs


def figures(layer,gemm):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):
        fig.savefig(FIGURES/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(FIGURES/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    methods=['ours_overall','ours_direct','neuralpower','gemmify_dt','gemmify_capacity','paleo_empirical']
    for target in ['standard','decomposed']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
        upper=1.2*max(r['mape_pct'] for r in layer if r['target']==target and r['layer'] in COMMON and r['method'] in methods)
        for ax,dtype in zip(axes,['fp16','fp32']):
            for j,m in enumerate(methods):
                values=[next(r['mape_pct'] for r in layer if (r['target'],r['dtype'],r['layer'],r['method'])==(target,dtype,l,m)) for l in COMMON]
                ax.plot(range(3),values,'o-',label=NAMES[m],color=COLORS[m],linewidth=1.5,markersize=5)
            ax.set_xticks(range(3),['Linear','Conv1x1','Conv3x3']); ax.set_yscale('log'); ax.grid(axis='y',alpha=.2)
            ax.set_title(dtype.upper());ax.set_ylim(4,upper)
        axes[0].set_ylabel('Test MAPE (%) - log scale')
        fig.suptitle(f'Layer latency: {target} path | validation-selected configurations')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.07))
        fig.tight_layout(rect=(0,.05,1,.94));finish(fig,'layer_'+target)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,dtype in zip(axes,['fp16','fp32']):
        ms=['ours_selected','neuralpower_fc_port','paleo_empirical']+(['neusight_bmm_clock','neusight_linear_clock'] if dtype=='fp32' else [])
        for m in ms:
            values=[next(r['mean_mape_pct'] for r in gemm if (r['dtype'],r['mode'],r['method'])==(dtype,mode,m)) for mode in ['allocate','prealloc','graph']]
            ax.plot(range(3),values,'o-',label=NAMES.get(m,'Ours: frozen GEMM'),color=COLORS[m],linewidth=1.5)
        ax.set_xticks(range(3),['Allocate','Preallocated','CUDA Graph']);ax.set_title(dtype.upper());ax.grid(axis='y',alpha=.2);ax.set_ylim(0,80)
    axes[0].set_ylabel('Test MAPE (%)');fig.suptitle('GEMM: 1,000 test shapes | 11,498 training shapes for local fits')
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.07))
    fig.tight_layout(rect=(0,.05,1,.94));finish(fig,'gemm_test')
    own=read(REFERENCE/'prepared_layer_full/aggregate_metrics.json')
    new=read(LAYER/'validation_aggregate.json')
    for target in ['standard','decomposed']:
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
        for row,dtype in enumerate(['fp16','fp32']):
            for col,l in enumerate(COMMON):
                ax=axes[row,col]
                for method in ['ours_direct','neuralpower','gemmify_dt','gemmify_capacity']:
                    candidates=[r for r in (own if method=='ours_direct' else new) if r['target']==target and r['dtype']==dtype and r['layer']==l
                                and r['method']==('direct' if method=='ours_direct' else method) and not r['failures']]
                    selected=[min(rs,key=lambda r:(r['mean_mape_pct'],r['mean_p90_pct'])) for rs in group(candidates,['k']).values()]
                    selected.sort(key=lambda r:r['k'])
                    ax.plot([r['actual_train_labels'] if method=='ours_direct' else r['mean_train_labels'] for r in selected],
                            [r['mean_mape_pct'] for r in selected],'o-',label=NAMES[method],color=COLORS[method],markersize=3)
                ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.2);ax.set_title(l+' '+dtype.upper())
                if row==1:ax.set_xlabel('Training labels per fit (includes pooled layers)')
                if col==0:ax.set_ylabel('Validation MAPE (%)')
        fig.suptitle(f'{target.capitalize()} path: observed validation learning curves')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,bbox_to_anchor=(.5,-.015))
        fig.tight_layout(rect=(0,.04,1,.95));finish(fig,'learning_'+target)


def main():
    TABLES.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
    verification=audit()
    layers,lp=layer_tables();gemm,gp=gemm_tables();budgets_rows,costs=budgets()
    figures(layers,gemm)
    save(HERE/'summary.json',dict(verification=verification,costs=costs,
        standard_common_six=[r for r in layers if r['target']=='standard' and r['layer'] in COMMON and r['method'] in ['ours_overall','neuralpower','gemmify_capacity']],
        layer_comparisons=lp,gemm_comparisons=gp,
        equivalent_accuracy_reached=[r for r in budgets_rows if r['goal']=='direct_mape' and r['reached']]))
    print('PASS: immutable inputs, saved models, metrics, selections and split separation verified')
    print('PASS: comparison tables, paired intervals and five figure pairs saved')


if __name__=='__main__':main()
