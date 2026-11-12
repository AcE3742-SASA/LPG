"""Verify frozen audit outputs and build the deadline comparison tables."""
from collections import defaultdict
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recheck as r
import report as old_report

c, a = r.c, r.a
TABLES = HERE/'tables'
LAYERS = ['linear', 'conv1', 'conv3']
METHODS = ['ours_direct_k200', 'neuralpower_fold_scaled', 'gemmify_dt', 'gemmify_capacity']
FIELDS = ['target', 'layer', 'dtype', 'method', 'k', 'seed']


def verify():
    original = c.read(r.OLD/'original_inputs.json')
    for name, digest in original.items():
        assert r.sha(r.ROOT/name) == digest, name
    old_manifest = c.read(r.OLD/'artifact_manifest.json')['artifacts']
    for name, digest in old_manifest.items():
        assert r.sha(r.OLD/name) == digest, name
    count = reloaded = 0
    dev = c.read(r.OLD/'development.json')
    cases = {(x['case']['group_id'], x['case']['layer']): x['case'] for x in dev}
    train = {(x['case']['group_id'], x['case']['layer'], x['dtype'], x['target']) for x in dev if x['case']['split']=='train'}
    valid = {(x['case']['group_id'], x['case']['layer'], x['dtype'], x['target']) for x in dev if x['case']['split']=='validation'}
    test = {(x['case']['group_id'], x['case']['layer'], x['dtype'], x['target']) for x in c.test_rows()}
    assert not (train & valid or train & test or valid & test)
    pipe = r.owner()
    for folder, fields, identity in [(HERE/'rerun', FIELDS, 'group_id'), (HERE/'gemm', ['dtype','mode','method','seed'], 'shape')]:
        r.verify(folder)
        for stage, prefix in [(folder, 'validation'), (folder/'evaluation', 'test')]:
            predictions = c.read(stage/(prefix+'_predictions.json'))
            groups = old_report.group(predictions, fields)
            metrics = c.read(stage/(prefix+'_metrics.json'))
            assert len(groups) == len(metrics)
            for metric in metrics:
                rows = groups[tuple(metric[f] for f in fields)]
                assert len({x[identity] for x in rows}) == len(rows) == metric['groups']
                score = c.score([x['actual_ms'] for x in rows], [x['predicted_ms'] for x in rows])
                assert score['failures'] == metric['failures'] == 0
                np.testing.assert_allclose([score[k] for k in ['mape_pct','p90_pct','rmspe_pct']],
                    [metric[k] for k in ['mape_pct','p90_pct','rmspe_pct']], rtol=1e-12, atol=1e-10)
                count += len(rows)
        refs = c.read(folder/('model_references.json' if folder.name=='rerun' else 'models.json'))
        val = old_report.group([x for x in c.read(folder/'validation_predictions.json') if x['method'] in METHODS], ['model_ref'])
        for ref in refs:
            model = joblib.load(folder/ref['model_ref'])
            assert r.sha(folder/ref['model_ref']) == ref['sha256']
            rows = val[ref['model_ref'],]
            if folder.name=='rerun':
                assert len(ref['train_keys']) == ref['train_labels']
                assert set(map(tuple, ref['train_keys'])) <= train
                xs = [cases[x['group_id'], x['layer']] for x in rows]
            else:
                xs = r.g.cases([tuple(map(int,x['shape'].split('x'))) for x in rows], 'validation')
            values = pipe.predict(model,xs,ref['dtype'],None) if ref['method']=='ours_direct_k200' else a.predict_fit(model,xs)
            np.testing.assert_allclose(values,[x['predicted_ms'] for x in rows],rtol=1e-12,atol=1e-12)
            reloaded += len(rows)
    candidates = old_report.group(c.read(HERE/'rerun/validation_aggregate.json'), ['target','layer','dtype','method'])
    for chosen in c.read(HERE/'rerun/selected.json'):
        rows = candidates[tuple(chosen[k] for k in ['target','layer','dtype','method'])]
        assert chosen == min((x for x in rows if not x['failures']),key=lambda x:(x['mean_mape_pct'],x['mean_p90_pct'],x['mean_train_labels'],x['k']))
    public = old_report.csv_read(HERE/'neusight_public_predictions.csv')
    grouped = old_report.group(public,['operator','device'])
    for metric in c.read(HERE/'neusight_public_audit.json')['metrics']:
        rows = grouped[metric['operator'],metric['device']]
        got = c.score([float(x['actual_ms']) for x in rows],[float(x['predicted_ms']) for x in rows])
        for field in ['groups','failures','mape_pct','p90_pct','rmspe_pct']:
            np.testing.assert_allclose(got[field],metric[field],rtol=1e-12,atol=1e-10)
    result = dict(status='PASS',original_inputs_unchanged=len(original),v1_artifacts_unchanged=len(old_manifest),
        new_predictions_recomputed=count,models_reloaded=222,validation_predictions_reproduced=reloaded,
        public_neusight_predictions_recomputed=len(public),split_intersections=0,
        interpretation='Post hoc descriptive comparison; percentile intervals do not correct for multiple comparisons')
    r.save(HERE/'verification.json', result)
    print('PASS: frozen files, predictions, model reloads and source data verified',result,flush=True)


def build_tables():
    new = c.read(HERE/'rerun/evaluation/test_aggregate.json')
    old = c.read(r.OLD/'runs/v1/evaluation/test_aggregate.json')
    fixed = [x for x in new+old if x['k']==200 and x['method'] in METHODS]
    assert len(fixed)==48
    c.csv_save(TABLES/'equal_pool_test.csv',fixed)
    pred = c.read(HERE/'rerun/evaluation/test_predictions.json')+c.read(r.OLD/'runs/v1/evaluation/test_predictions.json')
    comparisons = []
    for row in fixed:
        if row['method']=='ours_direct_k200': continue
        own = [x for x in pred if x['method']=='ours_direct_k200' and all(x[k]==row[k] for k in ['target','layer','dtype','k'])]
        other = [x for x in pred if all(x[k]==row[k] for k in ['target','layer','dtype','method','k'])]
        comparisons.append(old_report.paired(own,other,'group_id',dict(target=row['target'],layer=row['layer'],dtype=row['dtype'],baseline=row['method'],reference='ours_direct_k200')))
    c.csv_save(TABLES/'equal_pool_paired_intervals.csv',comparisons)
    refs = c.read(HERE/'rerun/model_references.json')+c.read(r.OLD/'runs/v1/model_references.json')
    pools = []
    for target in ['standard','decomposed']:
        for dtype in ['fp16','fp32']:
            for seed in [20261104,20261105,20261106]:
                sets = {}
                for method in METHODS:
                    sets[method] = {tuple(key) for x in refs if x['k']==200 and (x['target'],x['dtype'],x['seed'],x['method'])==(target,dtype,seed,method) for key in x['train_keys']}
                assert all(s==sets[METHODS[0]] for s in sets.values())
                assert all(len(s)==600 for s in sets.values())
                pools.append(dict(target=target,dtype=dtype,seed=seed,training_pool_labels=600,exact_same_training_keys=True))
    c.csv_save(TABLES/'equal_pool_identity.csv',pools)
    old_np = {(x['target'],x['layer'],x['dtype']):x for x in old if x['method']=='neuralpower' and x['selected_on_validation']}
    changes = []
    for row in new:
        if row['method']!='neuralpower_fold_scaled' or not row['selected_on_validation']: continue
        before = old_np[tuple(row[k] for k in ['target','layer','dtype'])]
        changes.append(dict(target=row['target'],layer=row['layer'],dtype=row['dtype'],old_k=before['k'],new_k=row['k'],
            old_mape_pct=before['mean_mape_pct'],new_mape_pct=row['mean_mape_pct'],delta_mape_pp=row['mean_mape_pct']-before['mean_mape_pct'],new_p90_pct=row['mean_p90_pct']))
    c.csv_save(TABLES/'neuralpower_selected_revision.csv',changes)
    direct = {(x['target'],x['layer'],x['dtype']):x for x in c.read(r.REFERENCE/'prepared_layer_full/selected_configurations.json') if x['selection_scope']=='direct'}
    candidates = [x for x in c.read(HERE/'rerun/validation_aggregate.json') if x['method']=='neuralpower_fold_scaled']
    dev = c.read(r.OLD/'development.json')
    def labels(row):
        family = 'fc' if row['layer']=='linear' else 'conv'
        return {tuple(key) for x in refs if (x['target'],x['dtype'],x['method'],x['k'],x['family'])==(row['target'],row['dtype'],row['method'],row['k'],family) for key in x['train_keys']}
    budgets = []
    for key, rows in old_report.group(candidates,['target','layer','dtype']).items():
        base = direct[key]
        goals = {'direct_mape':(base['mean_mape_pct'],float('inf')),'direct_mape_and_p90':(base['mean_mape_pct'],base['mean_p90_pct']),
                 'mape_10':(10,float('inf')),'mape_15':(15,float('inf')),'mape_20':(20,float('inf')),'legacy_15_30':(15,30)}
        nval = sum((x['target'],x['case']['layer'],x['dtype'])==key and x['case']['split']=='validation' for x in dev)
        for goal,(mape,p90) in goals.items():
            eligible=[x for x in rows if not x['failures'] and x['mean_mape_pct']<=mape+1e-12 and x['mean_p90_pct']<=p90+1e-12]
            entry=dict(target=key[0],layer=key[1],dtype=key[2],goal=goal,target_mape_pct=mape,target_p90_pct=p90 if np.isfinite(p90) else None,reached=bool(eligible))
            if eligible:
                best=min(eligible,key=lambda x:(len(labels(x)),x['mean_mape_pct'],x['mean_p90_pct'],x['k']))
                entry.update(k=best['k'],train_labels_per_model=best['mean_train_labels'],training_union_three_seeds=len(labels(best)),validation_labels=nval,
                    recipe_development_union=len(labels(best))+nval,validation_mape_pct=best['mean_mape_pct'],validation_p90_pct=best['mean_p90_pct'])
            budgets.append(entry)
    c.csv_save(TABLES/'neuralpower_equivalent_accuracy.csv',budgets)
    own_gemm = [x for x in old_report.csv_read(r.OLD/'tables/gemm_test.csv') if x['method']=='ours_selected']
    new_gemm = [x for x in c.read(HERE/'gemm/evaluation/test_aggregate.json') if x['method']=='neuralpower_fold_scaled']
    c.csv_save(TABLES/'gemm_revised.csv',own_gemm+new_gemm)
    ownpred=[x for x in old_report.csv_read(r.ROOT/'analysis/final_test_2026-11-11/gemm_expanded/predictions.csv') if x['variant']=='selected']
    otherpred=c.read(HERE/'gemm/evaluation/test_predictions.json')
    c.csv_save(TABLES/'gemm_revised_paired_intervals.csv',[
        old_report.paired([x for x in ownpred if (x['dtype'],x['mode'])==(row['dtype'],row['mode'])],
            [x for x in otherpred if all(x[k]==row[k] for k in ['dtype','mode','method'])], 'shape',dict(dtype=row['dtype'],mode=row['mode'],baseline=row['method'],reference='ours_selected')) for row in new_gemm])
    ownlayer=[x for x in old_report.csv_read(r.ROOT/'analysis/final_test_2026-11-11/layers/test_predictions.csv') if x['selection_scope']=='overall']
    chosen=[x for x in new if x['method']=='neuralpower_fold_scaled' and x['selected_on_validation']]
    c.csv_save(TABLES/'overall_vs_revised_np_intervals.csv',[
        old_report.paired([x for x in ownlayer if all(x[k]==row[k] for k in ['target','layer','dtype'])],
            [x for x in pred if all(x[k]==row[k] for k in ['target','layer','dtype','method','k'])], 'group_id',dict(target=row['target'],layer=row['layer'],dtype=row['dtype'],baseline=row['method'],reference='ours_overall')) for row in chosen])
    return fixed,comparisons,budgets


def figure(fixed):
    fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
    positions=np.arange(6)
    keys=[(layer,dtype) for layer in LAYERS for dtype in ['fp16','fp32']]
    for i,(method,title,color) in enumerate(zip([METHODS[0],METHODS[1],METHODS[3]],
            ['Local direct','NeuralPower-based port','GEMMify capacity variant'],['#225ea8','#d95f0e','#7b4fa3'])):
        values=[next(x['mean_mape_pct'] for x in fixed if (x['target'],x['layer'],x['dtype'],x['method'])==('standard',layer,dtype,method)) for layer,dtype in keys]
        bars=ax.bar(positions+(i-1)*.25,values,.24,label=title,color=color)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    ax.set_xticks(positions,[f'{layer.title()}\n{dtype.upper()}' for layer,dtype in keys])
    ax.set_ylim(0,53);ax.set_ylabel('Test MAPE (%)');ax.legend(frameon=False,ncol=3,loc='upper center',fontsize=9)
    ax.set_title('Standard layers: identical 600-label training pool',pad=15)
    ax.spines[['top','right']].set_visible(False);ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(folder/f'equal_pool_standard.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    TABLES.mkdir(exist_ok=True)
    verify()
    fixed,intervals,budgets=build_tables()
    figure(fixed)
    print('PASS: equal-pool comparison, corrected label budgets and figures saved',flush=True)
    for row in intervals:
        if row['baseline']=='neuralpower_fold_scaled':print(row)
    for row in budgets:
        if row['goal']=='direct_mape' and row['reached']:print('Equivalent accuracy:',row)
