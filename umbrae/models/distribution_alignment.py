"""Train-statistics-only analytic distribution transforms for P8.5."""
import numpy as np

def finalize_moments(count, total, cross):
    mean=total/count;cov=(cross-count*np.outer(mean,mean))/(count-1);cov=(cov+cov.T)*.5
    var=np.maximum(np.diag(cov),0);return mean,np.sqrt(var),cov

def symmetric_power(cov,power,epsilon=1e-4):
    values,vectors=np.linalg.eigh((cov+cov.T)*.5);clamped=np.maximum(values,epsilon);matrix=(vectors*clamped**power)@vectors.T
    meta={"smallest_eigenvalue":float(values.min()),"largest_eigenvalue":float(values.max()),"condition_number_before_clamp":float(values.max()/max(values.min(),np.finfo(float).tiny)),"clamped_dimension_count":int((values<epsilon).sum()),"epsilon_cov":epsilon};return matrix,meta

def fit_transforms(mean,std,cov,clip_mean,clip_std,clip_cov,epsilon=1e-4):
    inv,source=symmetric_power(cov,-.5,epsilon);color,target=symmetric_power(clip_cov,.5,epsilon)
    return {"mean_shift":clip_mean-mean,"affine_scale":clip_std/np.maximum(std,1e-8),"wct_matrix":inv@color,"source_eigensystem":source,"target_eigensystem":target}

def apply_transform(tokens,kind,mean,std,clip_mean,clip_std,wct=None):
    x=np.asarray(tokens,dtype=np.float32)
    if kind=='raw':return x
    if kind=='mean':result=x-mean+clip_mean
    elif kind=='affine':result=(x-mean)/np.maximum(std,1e-8)*clip_std+clip_mean
    elif kind=='wct':result=(x-mean)@wct+clip_mean
    else:raise ValueError(kind)
    return np.asarray(result,dtype=np.float32)
