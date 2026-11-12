"""Synthetic checks, including reuse of the original locked-label boundary tests."""
import tempfile
from pathlib import Path

import joblib
import numpy as np

import compare_learners as c
import test_rq14 as original_checks


def main():
    original_checks.test_label_boundary()
    original_checks.test_shape_grouping()
    original_checks.test_models_and_failures()
    rng=np.random.default_rng(42)
    x=rng.normal(size=(100,15)); y=.2*x[:,0]-.1*x[:,1]
    for learner in c.LEARNERS:
        model,detail=c.fit(learner,x,y,42)
        assert np.isfinite(model.predict(x)).all()
        assert detail['convergence_warnings']==0
        if learner.startswith('mlp'):
            assert np.allclose(model.regressor_.named_steps['standardscaler'].mean_,x.mean(axis=0))
            assert np.isclose(model.transformer_.mean_[0],y.mean())
            assert detail['parameter_count']==(1601 if learner=='mlp32' else 5249)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib';joblib.dump(model,path)
            assert np.array_equal(model.predict(x),joblib.load(path).predict(x))
        if learner=='rf':
            reference=c.old.RandomForestRegressor(**c.old.RF,random_state=42).fit(x,y)
            assert np.array_equal(model.predict(x),reference.predict(x))
    print('PASS: synthetic label boundary, group split, scaler, parameter, RF parity and serialization checks')


if __name__=='__main__':
    main()
