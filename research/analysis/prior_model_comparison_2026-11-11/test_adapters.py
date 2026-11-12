import importlib
from pathlib import Path
import unittest

import numpy as np

import adapters as a


def fc(rows=16, inputs=32, outputs=24):
    return dict(id='fc',group_id='fc',split='train',layer='linear',rows=rows,input_width=inputs,output_width=outputs)


def conv(kernel=3):
    return dict(id='conv',group_id='conv',split='train',layer='conv'+str(kernel),
                batch=2,height=7,width=9,cin=8,cout=16,kernel=kernel)


class PriorChecks(unittest.TestCase):
    def test_neuralpower_matches_author_prediction(self):
        source=(a.HERE/'sources/neuralpower/predict_runtime_power.py').read_text()
        start=source.index('def predict_runtime_power(')
        end=source.find('\ndef ',start+5)
        if end<0:end=source.index('\nif __name__',start)
        scope={'np':np}
        exec(source[start:end],scope)
        for kind,c,p,power_len in [('fc',fc(),[16,24,1,1,32],66),
                ('conv',conv(),[2,7,9,16,3,3,8,16,1,1,1,1,2,7,9,8],128)]:
            x=a.neuralpower_features([c])[0]
            weights=np.arange(1,len(x)+2,dtype=float)
            coeff={(kind,'runtime'):weights,(kind,'power'):np.zeros(power_len)}
            expected,_=scope['predict_runtime_power'](kind,p,coeff)
            self.assertEqual(float(np.r_[x,1]@weights),expected)
            self.assertEqual(len(x),21 if kind=='fc' else 127)

    def test_gemm_dimensions_preserve_conv_work(self):
        c=conv()
        m,n,k=a.gemm_shape(c)
        self.assertEqual(2*m*n*k,2*c['batch']*c['height']*c['width']*c['cout']*c['cin']*c['kernel']**2)

    def test_fit_rejects_validation_and_duplicates(self):
        for function in [a.fit_neuralpower,a.fit_gemmify]:
            first,second=fc(),dict(fc(),id='other',group_id='other',split='validation')
            with self.assertRaises(ValueError):function([first,second],np.array([1.,2.]),1)
            with self.assertRaises(ValueError):function([first,first],np.array([1.,2.]),1)

    def test_paleo_units_and_precision(self):
        c=fc(128,64,32)
        expected32=2*128*64*32/6.14e12*1000+4*(128*64+64*32+128*32)/262.3e9*1000
        self.assertAlmostEqual(a.paleo([c],'fp32')[0],expected32,places=12)
        expected16=2*128*64*32/35.02e12*1000+2*(128*64+64*32+128*32)/262.3e9*1000
        self.assertAlmostEqual(a.paleo([c],'fp16')[0],expected16,places=12)
        self.assertTrue(np.all(a.paleo([conv(1),conv(3)],'fp32')>0))

    def test_neusight_batch_matches_original_lookup_and_forward(self):
        neo=a.NeuSight()
        Meta=importlib.import_module('neusight.Model.meta').MetaTable
        torch=neo.torch
        for kind,shapes in [('LINEAR',[(16,24,32),(128,64,256)]),('BMM',[(16,24,32),(128,64,256)]),('VEC',[(16,32),(128,256)])]:
            got,detail=neo.predict(shapes,kind)
            model=neo.models[kind]
            table=Meta.__new__(Meta)
            table.features=model.features;table.df=neo.tables[kind]
            expected=[]
            for s in shapes:
                point=dict(Num_Sm=48,SingleFLOPs=8847.36,Dev_Mem=20,Mem_Bw=272.04,L2Cache=40)
                point.update(dict(B=1,M=s[0],N=s[2],K=s[1]) if kind!='VEC' else dict(B=s[0],H=s[1],OpsPerO=1,MemPerO=8))
                x=torch.tensor([[point[n] for n in model.features]],dtype=torch.float32)
                op='gelu' if kind=='VEC' else kind.lower()
                tiles=table.get_tile(x,None,[op]).reshape(1,2)
                with torch.inference_mode():expected.append(model([op],x,tiles=tiles).item()*1000)
            np.testing.assert_allclose(got,expected,rtol=2e-5,atol=1e-8)
            self.assertEqual(len(detail['matched_indices']),2)


if __name__=='__main__':unittest.main()
