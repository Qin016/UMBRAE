import numpy as np
from models.distribution_alignment import apply_transform,fit_transforms
def test_analytic_alignment_matches_first_two_moments():
 rng=np.random.default_rng(3);x=rng.normal(size=(20000,3))*[1,2,3]+[4,5,6];y=rng.normal(size=(20000,3))*[2,1,.5]+[-1,2,1];mx,sx=x.mean(0),x.std(0,ddof=1);my,sy=y.mean(0),y.std(0,ddof=1);cx=np.cov(x,rowvar=False);cy=np.cov(y,rowvar=False);t=fit_transforms(mx,sx,cx,my,sy,cy)
 a=apply_transform(x,'affine',mx,sx,my,sy);w=apply_transform(x,'wct',mx,sx,my,sy,t['wct_matrix']);assert np.allclose(a.mean(0),my);assert np.allclose(a.std(0,ddof=1),sy);assert np.allclose(np.cov(w,rowvar=False),cy,atol=2e-3)
