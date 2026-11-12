"""Small real-Ridge replay check; synthetic labels only."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import joblib
import numpy as np

import summarize_comparison as s


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'fit';out.mkdir();(out/'frozen_protocol.json').write_text('{}\n')
        train_shapes=[(64+16*i,80,96) for i in range(7)];new=[(400+16*i,80,96) for i in range(9)]
        vv={'legacy':[(200,80,96),(256,80,96)],'expansion':[(512,80,96),(640,80,96)]}
        train={(shape,'fp16','allocate'):.01+.000001*np.prod(shape) for shape in train_shapes+new}
        p=dict(added_sizes=[0,1,9],old_train_orders={str(seed):train_shapes for seed in s.c.old.SEEDS},
            new_train_orders={str(seed):new for seed in s.c.old.SEEDS},legacy_validation_shapes=vv['legacy'],new_validation_shapes=vv['expansion'])
        selected=[dict(learner='ridge',target=target,dtype='fp16',mode='allocate',added='1',n='8',validation='pooled') for target in ['direct','normalized']]
        predicted=[]
        for chosen in selected:
            for seed in s.c.old.SEEDS:
                x,l=s.c.old.design(train_shapes+new[:1],'fp16');y=s.c.old.target(train,train_shapes+new[:1],'fp16','allocate')
                response=np.log(y)-(np.log(l) if chosen['target']=='normalized' else 0)
                model,_=s.c.learners.fit('ridge',x,response,seed)
                for split,shapes in vv.items():
                    xv,lv=s.c.old.design(shapes,'fp16');pred=np.exp(model.predict(xv))*(lv if chosen['target']=='normalized' else 1)
                    predicted.extend({**{k:chosen[k] for k in s.FIELDS[:-1]},'seed':seed,'validation':split,'shape':s.c.old.shape_id(shape),'actual_ms':.1,'predicted_ms':value} for shape,value in zip(shapes,pred))
        s.c.old.save_csv(out/'validation_predictions.csv',predicted)
        with patch.object(s,'OUT',out),patch.object(s.c,'ROOT',root),patch.object(s.c,'source_labels',return_value=(train,{},[dict(split='synthetic')])):
            count=s.save_accuracy_weights(selected,p)
        assert count==6 and len(s.rows(out/'accuracy_first_model_refs.csv'))==6
        for path in (out/'accuracy_models').glob('*.joblib'):
            e=joblib.load(path);assert e['n']==8 and e['added']==1 and e['locked_test_opened'] is False
        assert json.loads((out/'accuracy_selection_label_access_log.json').read_text())==[dict(split='synthetic')]
    print('PASS: exact intermediate direct/normalized model replay, integer metadata, trusted references and synthetic-only labels')


if __name__=='__main__':main()
