"""Measured 2D screen-to-camera forward transport for a *fixed* optical rig.

Not a volumetric scattering model. The learned transfer is a ridge regression
on controlled 8x8 screen probes, evaluated on held-out physical camera frames.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


def screen_size():
    try:
        import ctypes
        if __import__('sys').platform.startswith('win'):
            user = ctypes.windll.user32
            user.SetProcessDPIAware()
            return user.GetSystemMetrics(0), user.GetSystemMetrics(1)
    except Exception:
        pass
    return 1280, 720


class Camera:
    """Use IP Webcam /shot.jpg, or a local OpenCV camera index."""
    def __init__(self, source: str):
        self.source = str(source)
        self.cap = None
        if self.source.strip().isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open local camera {self.source}')

    def frame(self):
        if self.cap is not None:
            # Flush a couple frames from a queued capture path.
            for _ in range(2):
                self.cap.grab()
            ok, img = self.cap.read()
            if not ok or img is None:
                raise RuntimeError('Camera read failed')
            return img
        url = self.source.rstrip('/')
        if not url.endswith('.jpg'):
            url += '/shot.jpg'
        sep = '&' if '?' in url else '?'
        with urlopen(f'{url}{sep}t={time.monotonic_ns()}', timeout=8) as resp:
            raw = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('IP Webcam returned an invalid JPEG')
        return img

    def close(self):
        if self.cap is not None:
            self.cap.release()


class Projector:
    """A black full-screen OpenCV projection, with no self-feedback preview."""
    def __init__(self, zone=420, width=None, height=None, fullscreen=True):
        sw, sh = screen_size()
        self.w, self.h = int(width or sw), int(height or sh)
        self.zone = min(int(zone), self.w - 80, self.h - 80)
        if self.zone < 96:
            raise ValueError('Projector screen or zone is too small')
        self.name = 'SplatYou / light window — Esc exits'
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.w, self.h)
        if fullscreen:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.black = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.last_key = -1

    def show(self, rgb: np.ndarray):
        frame = self.black.copy()
        px = self.zone
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        target = cv2.resize(np.uint8(np.clip(rgb, 0, 255)), (px, px),
                            interpolation=cv2.INTER_NEAREST if rgb.shape[0] <= 24 else cv2.INTER_CUBIC)
        x, y = (self.w - px)//2, (self.h - px)//2
        frame[y:y+px, x:x+px] = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.name, frame)
        self.last_key = cv2.waitKey(1) & 0xff
        return self.last_key

    def guide(self, message: str):
        frame = self.black.copy()
        for j, line in enumerate(message.split('\n')):
            cv2.putText(frame, line[:85], (max(20, self.w//15), max(60,self.h//4) + j*42),
                        cv2.FONT_HERSHEY_SIMPLEX, .79, (210, 222, 236), 2, cv2.LINE_AA)
        cv2.imshow(self.name, frame)
        while True:
            k = cv2.waitKey(30) & 255
            if k in (27, ord('q')):
                raise KeyboardInterrupt('Cancelled by user')
            if k in (32, 13):
                return

    def capture(self, camera: Camera, rgb: np.ndarray, settle=.6, samples=1):
        k = self.show(rgb)
        if k == 27:
            raise KeyboardInterrupt('Cancelled')
        time.sleep(max(0, settle))
        frames = [camera.frame() for _ in range(max(1,int(samples)))]
        if len({f.shape for f in frames}) != 1:
            raise RuntimeError('Camera resolution changed while measuring')
        return np.mean(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def close(self):
        cv2.destroyWindow(self.name)


def get_roi(black: np.ndarray, white: np.ndarray):
    if black.shape != white.shape:
        raise ValueError('Camera size changed during reference capture')
    b = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY).astype(np.float32)
    change = np.maximum(0, w - b)
    high = float(np.percentile(change, 99.5))
    if high < 5.0:
        raise RuntimeError(f'Only {high:.2f}/255 white-minus-black signal. Increase manual exposure or thin the diffuser.')
    smooth = cv2.GaussianBlur(change, (0,0), 6)
    mask = (smooth > max(3, high*.12)).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise RuntimeError('Could not find the illuminated region')
    k = 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    x,y,wid,hei,_ = map(int,stats[k])
    pad = max(8,int(max(wid,hei)*.1))
    x0,y0=max(0,x-pad),max(0,y-pad)
    x1,y1=min(black.shape[1],x+wid+pad),min(black.shape[0],y+hei+pad)
    return (x0,y0,x1-x0,y1-y0), high


def camera_features(frame: np.ndarray, roi, side=14):
    x,y,w,h = [int(k) for k in roi]
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        raise ValueError('Camera region is empty')
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255


def screen_features(rgb: np.ndarray, grid=8):
    if rgb.ndim == 3:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(np.asarray(rgb,np.float32), (grid, grid),
                      interpolation=cv2.INTER_AREA).ravel()/255


def pattern(rng: np.random.Generator, grid=8, index=0):
    kind = index % 4
    if kind == 0:
        x = (rng.random((grid, grid)) > .5).astype(np.uint8)*255
    elif kind == 1:
        small = rng.integers(0,2,(max(2,grid//2),max(2,grid//2)),dtype=np.uint8)*255
        x = cv2.resize(small,(grid,grid),interpolation=cv2.INTER_NEAREST)
    elif kind == 2:
        x = np.zeros((grid, grid),np.uint8)
        x[:,rng.integers(0,grid):] = 255
        if index%8==2: x = x.T.copy()
    else:
        x = np.zeros((grid,grid),np.uint8)
        a,b = rng.integers(0,max(1,grid-2),2)
        x[b:min(grid,b+max(2,grid//2)), a:min(grid,a+max(2,grid//2))]=255
    return np.repeat(x[:,:,None],3,axis=2)


class MeasuredForward:
    """y = y_mean + (x-x_mean) @ W; features, not radiometric physics."""
    def __init__(self, grid=8, side=14, ridge=.8):
        self.grid=int(grid); self.side=int(side); self.ridge=float(ridge)
        self.roi=None; self.W=None; self.xmean=None; self.ymean=None
        self.metrics={}

    def fit(self, x, y, *, holdout=.16):
        x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<12:
            raise ValueError('Expected >=12 paired screen/camera measurements')
        ntrain = max(8,int(len(x)*(1-holdout)))
        # Deterministic contiguous test tail: all test frames were withheld.
        xt,yt=x[:ntrain],y[:ntrain]
        self.xmean=xt.mean(axis=0);self.ymean=yt.mean(axis=0)
        xc=xt-self.xmean
        reg = np.eye(xc.shape[1])*self.ridge
        self.W=np.linalg.solve(xc.T@xc+reg,xc.T@(yt-self.ymean))
        prediction=self.predict(x[ntrain:])
        error = float(np.sqrt(np.mean((prediction-y[ntrain:])**2)))
        constant = float(np.sqrt(np.mean((self.ymean[None,:]-y[ntrain:])**2)))
        total = float(np.sqrt(np.mean((yt-self.ymean)**2)))
        self.metrics={"test_rmse":error,"constant_baseline_rmse":constant,
                      "train_camera_std":total,"heldout":int(len(x)-ntrain),
                      "trained":int(ntrain),"signal_ratio":round(float(constant/(error+1e-10)),4),
                      "forward_pass":bool(error<constant*.9)}
        return self.metrics

    def predict(self, x):
        a=np.asarray(x,np.float64)
        if a.ndim==1: a=a[None]
        return (a-self.xmean)@self.W+self.ymean

    def observe(self, frame):
        return camera_features(frame,self.roi,self.side)

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,W=self.W,xmean=self.xmean,ymean=self.ymean,
            roi=self.roi,grid=self.grid,side=self.side,ridge=self.ridge,metrics=json.dumps(self.metrics))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as o:
            f=cls(int(o['grid']),int(o['side']),float(o['ridge']))
            f.W=o['W'];f.xmean=o['xmean'];f.ymean=o['ymean'];f.roi=tuple(o['roi'])
            f.metrics=json.loads(str(o['metrics']))
        return f


def acquire(projector, camera, decoder, *, grid=8, side=14, count=120,
            settle=.6, samples=1, output='runs/rig'):
    """Physical calibration with a fixed camera/display/diffuser. No target leakage."""
    rng=np.random.default_rng(20261119)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    projector.guide('SPLATYOU / PHYSICAL OPTICAL CALIBRATION\n\nLock focus, exposure, white balance and screen brightness.\n\nKeep camera, screen and diffuser fixed.\n\nClose any IP Webcam preview on the monitor.\n\nSPACE start / ESC cancel')
    black=np.zeros((grid,grid,3),np.uint8);white=np.full_like(black,230)
    b=projector.capture(camera,black,settle,samples)
    w=projector.capture(camera,white,settle,samples)
    roi,range_peak=get_roi(b,w)
    cv2.imwrite(str(out/'ref_black.jpg'),b)
    cv2.imwrite(str(out/'ref_white.jpg'),w)
    f=MeasuredForward(grid,side);f.roi=roi
    print(f'Camera ROI: {roi}; peak white-minus-black: {range_peak:.1f}/255')
    X=[];Y=[]
    for i in range(int(count)):
        # Include actual decoder outputs so the transfer sees the face tone range.
        if i%5==4:
            z = __import__('splatyou.model',fromlist=['new_latent']).new_latent(rng,radius=float(rng.choice([7.,10.,15.])))
            rgb=np.uint8(decoder(z)[0]*230)
        else:
            rgb=pattern(rng,grid,i)
        frame=projector.capture(camera,rgb,settle,samples)
        X.append(screen_features(rgb,grid));Y.append(f.observe(frame))
        if (i+1)%10==0 or i==0:
            print(f'Optical pairs: {i+1}/{count}',flush=True)
    # Repeat the identical black/white inputs AFTER the sequence, rather than
    # assuming the phone stayed locked throughout 120 measurements.
    b_again=projector.capture(camera,black,settle,samples)
    w_again=projector.capture(camera,white,settle,samples)
    early_b=f.observe(b);early_w=f.observe(w)
    late_b=f.observe(b_again);late_w=f.observe(w_again)
    response_rms=float(np.sqrt(np.mean((early_w-early_b)**2)))
    drift_rms=max(float(np.sqrt(np.mean((late_b-early_b)**2))),
                  float(np.sqrt(np.mean((late_w-early_w)**2))))
    drift_fraction=drift_rms/max(response_rms,1e-5)
    x0,y0,rw,rh=roi
    clipping=float(np.mean(w[y0:y0+rh,x0:x0+rw]>=252))
    stable=bool(drift_fraction<.08 and clipping<.005)
    X=np.asarray(X);Y=np.asarray(Y)
    m=f.fit(X,Y)
    m.update(camera_stability_pass=stable,reference_drift_fraction=round(drift_fraction,5),
             white_clipped_fraction=round(clipping,5))
    if not stable:
        print('WARNING: black/white repeatability or clipping FAILED. Camera or rig likely drifted.')
    f.save(out/'forward.npz')
    np.savez_compressed(out/'paired_data.npz',X=X,Y=Y)
    (out/'report.json').write_text(json.dumps(dict(m,roi=roi,white_minus_black=range_peak),indent=2))
    print('Held-out measured forward RMSE:',round(m['test_rmse'],5),
          'mean-image baseline:',round(m['constant_baseline_rmse'],5),
          'pass:',m['forward_pass'])
    return f
