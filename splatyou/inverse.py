"""Measured forward + compact face-manifold inverse with optional active masks.

The unknown target latent is never passed to the solver. A mismatch flag is
reported when the phone measurement cannot be explained within calibrated
forward-model error. SplatWorld4's weak-direction active inquiry inspired the
SVD probe choice; this is NOT reuse of SplatWorld4's trained world model.
"""
from __future__ import annotations
import numpy as np
import cv2
from .model import new_latent, local_basis
from .optics import screen_features


def mask_bank(grid=8):
    yy,xx=np.mgrid[0:grid,0:grid]
    out=[('whole',np.ones((grid,grid),np.float32)),
         ('left',(xx<grid//2).astype(np.float32)),
         ('right',(xx>=grid//2).astype(np.float32)),
         ('upper',(yy<grid//2).astype(np.float32)),
         ('lower',(yy>=grid//2).astype(np.float32)),
         ('checker',((xx//2+yy//2)%2==0).astype(np.float32)),
         ('anti-checker',((xx//2+yy//2)%2==1).astype(np.float32))]
    return out


def masked_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h,w=rgb.shape[:2]
    m=cv2.resize(np.asarray(mask,np.float32),(w,h),interpolation=cv2.INTER_NEAREST)
    return np.asarray(rgb,np.float32)*m[...,None]


def estimate_observations(decoder, forward, zs, masks, batch=24):
    """Outputs (n-latents, n-probes*camera-feature-count)."""
    zs=np.asarray(zs,np.float32)
    if zs.ndim==1:zs=zs[None]
    results=[]
    for p in range(0,len(zs),batch):
        rgb=decoder(zs[p:p+batch])
        xs=[]
        for mask in masks:
            m=np.asarray(mask,np.float32)
            # Grayscale downsampling and mask should be applied before spatial
            # integration. cv2 resize preserves the camera-controlled pattern.
            xx=np.stack([screen_features(masked_image(im,m)*255,forward.grid) for im in rgb])
            xs.append(forward.predict(xx))
        results.append(np.concatenate(xs,axis=1))
    return np.concatenate(results,axis=0)


def damped_step(J, residual, damping=1e-4, trust=2.):
    J=np.asarray(J,np.float64);r=np.asarray(residual,np.float64)
    u,s,v=np.linalg.svd(J,full_matrices=False)
    reg=float(damping)*max(1e-10,float(np.mean(s*s)))
    dz=v.T@((s/(s*s+reg))*(u.T@r))
    norm=np.linalg.norm(dz)
    return dz*min(1.,float(trust)/(norm+1e-12))


def solve_latent(decoder, forward, measured, masks, *, seed=19,
                 candidates=128, top=3, dims=8, steps=10,
                 z_starts=None):
    """Return the best observed-fit latent and search diagnostics.

    No truth latent nor true screen pixels enter this function. Optimizes the
    128-D decoder only in locally measured 8D tangent charts + radial axis.
    """
    masks=[m for m in masks]
    measured=np.asarray(measured,np.float64).reshape(-1)
    rng=np.random.default_rng(seed)
    starts=[] if z_starts is None else [np.asarray(z,np.float32) for z in z_starts]
    starts.extend(new_latent(rng,float(rng.choice([5.,8.,11.,16.,24.]))) for _ in range(int(candidates)))
    zs=np.stack(starts)
    preds=estimate_observations(decoder,forward,zs,masks)
    losses=np.mean((preds-measured[None])**2,axis=1)
    best_idxs=np.argsort(losses)[:max(1,int(top))]
    before=float(np.sqrt(np.min(losses)))
    best_z=zs[best_idxs[0]].copy()
    best_error=before
    history=[]
    for seed_id,idx in enumerate(best_idxs):
        z=zs[idx].copy()
        base=local_basis(z,dimensions=dims,seed=seed+seed_id+100)
        radial=z/(np.linalg.norm(z)+1e-10)
        B=np.concatenate([base,radial[None,:]],axis=0)
        cur=float(np.sqrt(losses[idx]))
        history.append([float(cur)])
        for _ in range(int(steps)):
            eps=.4
            variants=np.concatenate([z[None]+eps*B, z[None]-eps*B],axis=0)
            projected=estimate_observations(decoder,forward,variants,masks)
            J=((projected[:len(B)]-projected[len(B):])/(2*eps)).T
            now=estimate_observations(decoder,forward,z,masks)[0]
            d=damped_step(J,measured-now,trust=2.0)
            # Restrict radius to meaningful decoder range; avoid diving into
            # arbitrary enormous 'fire' where a bad forward fit could cheat.
            improved=False
            for fraction in (1.,.5,.25):
                new=z+(fraction*d).astype(np.float32)@B
                norm=np.linalg.norm(new)
                if norm>37: new=new*(37/norm)
                if norm<2: new=new*(2/max(norm,1e-8))
                y=estimate_observations(decoder,forward,new,masks)[0]
                score=float(np.sqrt(np.mean((y-measured)**2)))
                if score < cur-1e-7:
                    cur=score;z=new.astype(np.float32);improved=True;break
            history[-1].append(cur)
            if not improved:
                break
        if cur<best_error:
            best_z=z.copy();best_error=cur
    # Candidate spread is *not* calibrated posterior probability.
    order=np.argsort(losses)
    return {'latent':best_z,'observed_rmse':best_error,'initial_rmse':before,
            'all_candidate_rmse':np.sqrt(losses[order[:12]]).tolist(),
            'histories':history,'radius':float(np.linalg.norm(best_z))}


def choose_next_mask(decoder, forward, z, already, options=None, dims=8):
    """SplatWorld4-style SVD null-space probe design with *no target peeking*.

    Probes are physical screen masks (not camera positions). The score uses only
    the current estimated latent, measured/calibrated forward and chosen masks.
    """
    options=options or mask_bank(forward.grid)
    B=np.concatenate([local_basis(z,dims,seed=77),
                      (z/(np.linalg.norm(z)+1e-10))[None]],axis=0)
    n=len(B);eps=.35
    candidates=np.concatenate([z[None]+eps*B,z[None]-eps*B])
    J_all=[]
    for name,m in options:
        pr=estimate_observations(decoder,forward,candidates,[m])
        J_all.append(((pr[:n]-pr[n:])/(2*eps)).T)
    observed=np.vstack([J_all[i] for i in already])
    # Full SVD is essential: compact SVD hides a nullspace when measured
    # camera features are fewer than latent dimensions.
    _u,s,V=np.linalg.svd(observed,full_matrices=True)
    threshold=s.max()*1e-3 if len(s) else 0
    rank=int(np.sum(s>threshold))
    weak=V[rank:].T if rank<n else V[-2:].T
    scores={i:float(np.sum((J@weak)**2)) for i,J in enumerate(J_all) if i not in already}
    if not scores:
        return None,{},rank
    return max(scores,key=lambda i:(scores[i],-i)),scores,rank


def direct_pixel_baseline(forward, observed, masks, alpha=.2):
    """Model-free regularized 8x8 estimate, for a fair reconstruction baseline."""
    blocks=[];rhs=[]
    for y,m in zip(observed,masks):
        matrix=m.reshape(-1)[:,None]*forward.W
        blocks.append(matrix.T)
        rhs.append(np.asarray(y)-forward.ymean+forward.xmean@forward.W)
    G=np.vstack(blocks);h=np.concatenate(rhs)
    mat=G.T@G+alpha*np.eye(forward.grid**2)
    return np.clip(np.linalg.solve(mat,G.T@h+alpha*.5),0,1).reshape(forward.grid,forward.grid)
