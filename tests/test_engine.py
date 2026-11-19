import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
from splatyou.model import local_basis,new_latent
from splatyou.residue import bands,compose,transport_anchor
from splatyou.optics import MeasuredForward,screen_features,pattern,get_roi
from splatyou.inverse import (mask_bank,masked_image,estimate_observations,
                               choose_next_mask,solve_latent,direct_pixel_baseline)
from splatyou import __version__


class SimpleDecoder:
    def __call__(self, zs):
        z=np.atleast_2d(zs).astype(np.float32)
        y,x=np.mgrid[0:96,0:96].astype(np.float32)
        b=np.stack([np.exp(-((x-i*11-16)**2+(y-i*9-15)**2)/(2*120.)) for i in range(6)])
        gray=np.clip(.4+np.einsum('ni,ihw->nhw',np.tanh(z[:,:6]),b)*.1,0,1)
        return np.repeat(gray[:,:,:,None],3,axis=3)


class TestFusion(unittest.TestCase):
    def test_sihti_telescoping_exact(self):
        rng=np.random.default_rng(5)
        x=rng.random((70,80,3),dtype=np.float32)
        c,r=bands(x)
        self.assertLess(float(np.max(np.abs(compose(c,r)-x))),2e-6)

    def test_no_anchor_cannot_add_detail(self):
        decoder=SimpleDecoder()
        x=decoder(np.zeros((1,128),dtype=np.float32))[0]
        a=transport_anchor(np.ones((96,96,3),np.float32)*.5,x,x,gain=0)
        self.assertLess(float(np.max(np.abs(x-a))),1e-6)

    def test_model_basis_is_tangent(self):
        z=new_latent(np.random.default_rng(4),8)
        B=local_basis(z,8)
        self.assertEqual(B.shape,(8,128))
        self.assertLess(float(np.max(np.abs(B@(z/np.linalg.norm(z))))),1e-5)

    def test_measured_forward_and_no_target_leak(self):
        rng=np.random.default_rng(10)
        x=rng.random((140,64))
        A=rng.normal(0,.025,(64,36))
        y=x@A+.25
        f=MeasuredForward(grid=8,side=6,ridge=.02)
        metrics=f.fit(x,y)
        self.assertTrue(metrics['forward_pass'])
        self.assertLess(metrics['test_rmse'],metrics['constant_baseline_rmse']*.15)
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/'forward.npz';f.roi=(2,3,8,8);f.save(file)
            reloaded=MeasuredForward.load(file)
            np.testing.assert_allclose(reloaded.predict(x[2]),f.predict(x[2]))
            self.assertEqual(reloaded.roi,f.roi)

    def test_roi_detects_real_difference(self):
        a=np.zeros((100,150,3),np.uint8)
        b=a.copy();b[20:70,35:100,:]=150
        roi, peak=get_roi(a,b)
        x,y,w,h=roi
        self.assertLessEqual(x,35); self.assertGreaterEqual(x+w,100)
        self.assertGreater(peak,140)

    def test_mask_selection_has_no_truth_argument(self):
        rng=np.random.default_rng(45)
        f=MeasuredForward(grid=8,side=6,ridge=.01)
        X=rng.random((125,64));Y=X@rng.normal(0,.05,(64,36))
        f.fit(X,Y);f.roi=(0,0,6,6)
        z=new_latent(rng,8)
        result,scores,rank=choose_next_mask(SimpleDecoder(),f,z,[0],dims=4)
        self.assertNotEqual(result,0)
        self.assertIn(result,scores)
        self.assertGreaterEqual(rank,0)
        self.assertLessEqual(rank,5)

    def test_latent_fit_improves_without_truth(self):
        rng=np.random.default_rng(67)
        f=MeasuredForward(grid=8,side=6,ridge=.02)
        X=rng.random((180,64));Y=X@rng.normal(0,.03,(64,36))+.3
        f.fit(X,Y)
        d=SimpleDecoder();z=new_latent(rng,8)
        masks=[mask_bank(8)[0][1],mask_bank(8)[2][1]]
        obs=estimate_observations(d,f,z,masks)[0]
        fitted=solve_latent(d,f,obs,masks,candidates=45,top=2,dims=7,steps=7)
        self.assertLess(fitted['observed_rmse'],fitted['initial_rmse']+.00000001)
        self.assertLess(fitted['observed_rmse'],.03)
        baseline=direct_pixel_baseline(f,[obs[:36],obs[36:]],masks)
        self.assertEqual(baseline.shape,(8,8))


if __name__=='__main__':
    unittest.main()
