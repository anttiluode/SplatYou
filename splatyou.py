#!/usr/bin/env python3
"""SplatYou: a genuine SplatWorld ONNX decoder in three compact optical loops.

studio   — continuous wave-face + frozen real-image Sihti-style residue
inverse  — learn physical screen->camera optics, fit face latent from camera only
feedback — model-predicted camera residual drives the decoder back onto screen
simulate — reproducible synthetic test, explicitly NOT physical validation

No training of a new SplatWorld decoder; the 7.2 MB ONNX stays unchanged.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from splatyou.model import (WaveDecoder,find_model,download_model,new_latent,
                           local_basis,image_u8,shell_name)
from splatyou.residue import bands,compose,transport_anchor
from splatyou.optics import (Camera,Projector,MeasuredForward,acquire,pattern,
                             screen_features,camera_features)
from splatyou.inverse import (mask_bank,masked_image,estimate_observations,
                              solve_latent,choose_next_mask,direct_pixel_baseline)


def decoder_for(args):
    path=find_model(args.model)
    if path is None and args.download_model:
        path=download_model('splat_decoder.onnx')
    if path is None:
        raise RuntimeError('Missing real SplatWorld ONNX. Put splat_decoder.onnx beside '
                           'splatyou.py, pass --model PATH, or use --download-model. '
                           'There is NO fake fallback in physical or studio modes.')
    print('Loading authentic SplatWorld decoder:',path)
    return WaveDecoder(path)


def put_text(img,lines,origin=(12,25),scale=.49):
    for i,line in enumerate(lines):
        cv2.putText(img,str(line),(origin[0],origin[1]+i*23),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,(218,236,232),1,cv2.LINE_AA)


def label_panel(img,label):
    t=np.zeros((34,img.shape[1],3),np.uint8)
    cv2.putText(t,label,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,228,239),1,cv2.LINE_AA)
    return np.concatenate([t,img],axis=0)


def studio(args,decoder):
    """Live local latent wave explorer. Real details require --anchor."""
    rng=np.random.default_rng(args.seed)
    anchor=None
    if args.anchor:
        src=cv2.imread(str(args.anchor))
        if src is None:raise FileNotFoundError(args.anchor)
        anchor=cv2.cvtColor(src,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        anchor=cv2.resize(anchor,(512,512),interpolation=cv2.INTER_AREA)
    z0=new_latent(rng,8.)
    B=local_basis(z0,2,seed=args.seed)
    name='SplatYou | studio : move mouse, SPACE new, A anchor, S save, Q exit'
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,1200,550)
    cv2.createTrackbar('radius x10',name,80,550,lambda _:None)
    cv2.createTrackbar('detail %',name,80,200,lambda _:None)
    state={'dx':0.,'dy':0.,'dragging':False,'x':0,'y':0,'enabled':True}
    def on_mouse(event,x,y,flags,_userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            state['dragging']=True;state['x']=x;state['y']=y
        elif event==cv2.EVENT_MOUSEMOVE and state['dragging']:
            state['dx']=float(np.clip(state['dx']+(x-state['x'])*.02,-12,12))
            state['dy']=float(np.clip(state['dy']+(y-state['y'])*.02,-12,12))
            state['x']=x;state['y']=y
        elif event==cv2.EVENT_LBUTTONUP:state['dragging']=False
    cv2.setMouseCallback(name,on_mouse)
    frozen=decoder(z0)[0]
    tick=0
    try:
        while True:
            radius=max(2.,cv2.getTrackbarPos('radius x10',name)/10.)
            gain=cv2.getTrackbarPos('detail %',name)/100.
            z=z0+B[0]*state['dx']+B[1]*state['dy']
            z=z*radius/(np.linalg.norm(z)+1e-10)
            face=decoder(z)[0]
            source=image_u8(face,380)
            if anchor is not None and state['enabled']:
                enriched=transport_anchor(anchor,frozen,face,gain)
                enriched=image_u8(enriched,380)
            else:
                enriched=source.copy()
            core,res=bands(cv2.resize(face,(380,380),interpolation=cv2.INTER_CUBIC))
            band_view=np.abs(res[0])+np.abs(res[1])*.7
            band_view=np.uint8(np.clip(band_view*950,0,255))
            if band_view.ndim==2:band_view=cv2.cvtColor(band_view,cv2.COLOR_GRAY2RGB)
            tiles=[label_panel(source,'RAW ONNX (96px)'),
                   label_panel(enriched,'ANCHOR RESIDUE' if anchor is not None else 'NO REAL ANCHOR'),
                   label_panel(band_view,'SIHTI-STYLE HIGH BANDS')]
            panel=cv2.hconcat(tiles)
            top=np.zeros((88,panel.shape[1],3),np.uint8)
            put_text(top,[f'|z| = {radius:.2f}  {shell_name(radius)}  (artistic region, NOT confidence)',
                          f'Anchor: {"real immutable image" if anchor is not None else "none: no extra detail invented"}   '
                          'Drag to morph  |  SPACE new identity  |  A toggle detail  |  S save  |  Q exit'],
                     (14,31),.53)
            display=cv2.cvtColor(np.concatenate([top,panel],axis=0),cv2.COLOR_RGB2BGR)
            cv2.imshow(name,display)
            key=cv2.waitKey(1)&255
            if key in (27,ord('q')):break
            if key==ord('a'):state['enabled']=not state['enabled']
            if key==ord('s'):
                folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(folder/f'studio_{tick:05d}.png'),display)
                print('Saved studio screenshot',flush=True)
            if key in (32,ord('n')):
                z0=new_latent(rng,8.);B=local_basis(z0,2,seed=int(rng.integers(1<<24)))
                frozen=decoder(z0)[0]
                state['dx']=state['dy']=0.
                # Anchor cannot be transported between independently picked identities
                # without re-pairing. Toggle is deliberately disabled on identity jump.
                if anchor is not None:
                    state['enabled']=False
                    print('New identity: frozen anchor residue disabled (press A to override)')
            tick+=1
            time.sleep(.015)
    finally:
        cv2.destroyWindow(name)


def start_rig(args,decoder):
    cam=Camera(args.camera)
    w,h=None,None
    if args.screen:
        try:w,h=map(int,args.screen.lower().split('x'))
        except Exception as exc:raise ValueError('--screen must be like 1920x1080') from exc
    projector=Projector(zone=args.zone,width=w,height=h)
    return cam,projector


def get_calibration(args,cam,projector,decoder):
    if args.calibration:
        f=MeasuredForward.load(args.calibration)
        print('Loaded calibration',args.calibration,'roi',f.roi)
        print('WARNING: calibration is valid only if camera, screen, exposure and diffuser have NOT changed.')
        return f
    return acquire(projector,cam,decoder,grid=args.grid,side=args.side,
                   count=args.patterns,settle=args.settle,samples=args.samples,
                   output=Path(args.output)/'calibration')


def rgb_from_file(path):
    x=cv2.imread(str(path))
    if x is None:raise FileNotFoundError(path)
    return cv2.cvtColor(cv2.resize(x,(96,96),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB).astype(np.float32)/255


def save_inverse_summary(out,target,fit,baseline,report,measured_frame):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    target_image=image_u8(target,384)
    got_image=image_u8(fit,384)
    baseline_image=cv2.resize(np.uint8(np.clip(baseline*255,0,255)),(384,384),interpolation=cv2.INTER_NEAREST)
    baseline_image=cv2.cvtColor(baseline_image,cv2.COLOR_GRAY2RGB)
    white=np.zeros((40,384,3),np.uint8)
    labels=['TARGET / evaluation only','LATENT PRIOR RECON','PIXEL-ONLY BASELINE']
    rows=[]
    for name,im in zip(labels,[target_image,got_image,baseline_image]):
        header=white.copy();put_text(header,[name],(10,27),.55)
        rows.append(np.concatenate([header,im]))
    triptych=np.concatenate(rows,axis=1)
    cv2.imwrite(str(out/'comparison.png'),cv2.cvtColor(triptych,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/'last_camera.jpg'),measured_frame)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    return triptych


def inverse(args,decoder):
    cam,proj=start_rig(args,decoder)
    try:
        f=get_calibration(args,cam,proj,decoder)
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        if args.target_image:
            truth=rgb_from_file(args.target_image)
            source='external image (OOD relative to SplatWorld face prior)'
        else:
            # This truth is used ONLY by the physical projector and final eval,
            # never by solve_latent() or choose_next_mask().
            truth=decoder(new_latent(np.random.default_rng(args.seed+555),8.))[0]
            source='held-out ONNX face'
        masks=mask_bank(f.grid)
        proj.guide('SPLATYOU / PHYSICAL INVERSE\n\nCalibrated. Keep everything fixed.\n\nA new unseen face/photo will appear behind diffuser.\n\nThe solver receives CAMERA readings only.\n\nSPACE: begin active light-mask queries')
        chosen=[0]; observations=[]; last_frame=None; results=[]
        for query_i in range(max(1,args.views)):
            index=chosen[-1]
            probe=masked_image(truth,masks[index][1])
            frame=proj.capture(cam,image_u8(probe),args.settle,args.samples)
            last_frame=frame
            observations.append(f.observe(frame))
            active=[masks[i][1] for i in chosen]
            print('Query',query_i+1,'/',args.views,':',masks[index][0],flush=True)
            solved=solve_latent(decoder,f,np.concatenate(observations),active,
                                seed=args.seed,candidates=args.candidates,
                                top=args.top,dims=args.dimensions,steps=args.steps)
            fit=decoder(solved['latent'])[0]
            baseline=direct_pixel_baseline(f,observations,active)
            image_rmse=float(np.sqrt(np.mean((fit-truth)**2)))
            direct_rmse=float(np.sqrt(np.mean((cv2.resize(baseline,(96,96),interpolation=cv2.INTER_CUBIC)[...,None]-truth)**2)))
            fit_noise=float(solved['observed_rmse'])
            status=('CAMERA REPEATABILITY FAILED' if not f.metrics.get('camera_stability_pass',True)
                    else 'MEASURED FORWARD CALIBRATION WEAK' if not f.metrics.get('forward_pass',False)
                    else 'POOR FIT / POSSIBLE MODEL MISMATCH' if fit_noise>max(.01,2.5*f.metrics['test_rmse'])
                    else 'OPTICAL FIT OK; IDENTITY MAY REMAIN AMBIGUOUS')
            entry=dict(query=query_i+1,mask=masks[index][0],status=status,
                       observation_rmse=fit_noise,initial_rmse=solved['initial_rmse'],
                       groundtruth_pixel_rmse=image_rmse,direct_grid_pixel_rmse=direct_rmse,
                       artistic_radius=solved['radius'])
            results.append(entry);print(json.dumps(entry,indent=2),flush=True)
            if query_i+1<args.views:
                nxt,scores,rank=choose_next_mask(decoder,f,solved['latent'],chosen,
                                                 options=masks,dims=args.dimensions)
                if nxt is None:break
                print('Selected NEW mask:',masks[nxt][0],'; observed local rank',rank)
                chosen.append(nxt)
        report=dict(mode='physical_active_inverse',source=source,calibration=f.metrics,
                    queries=results,chosen_masks=[masks[i][0] for i in chosen],
                    claim='2D measured optical forward + face-family prior; NOT 3D CT or identity proof')
        view=save_inverse_summary(out,truth,fit,baseline,report,last_frame)
        proj.close()
        name='SplatYou: CAMERA-ONLY reconstruction and blinded evaluation'
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name,cv2.cvtColor(view,cv2.COLOR_RGB2BGR))
        print('Saved',out/'comparison.png','and report.json')
        print('Press any key in results window to exit.')
        cv2.waitKey(0)
        cv2.destroyWindow(name)
    finally:
        try:proj.close()
        except Exception:pass
        cam.close()


def feedback(args,decoder):
    cam,proj=start_rig(args,decoder)
    try:
        f=get_calibration(args,cam,proj,decoder)
        proj.guide('SPLATYOU / OPTICAL FACE RESONATOR\n\nMove your hand or a glass into the light path.\n\nPhysical residual -> local face directions + artistic radius.\n\nNo fake hallucination confidence.\n\nSPACE: start feedback / ESC: cancel')
        rng=np.random.default_rng(args.seed)
        z0=new_latent(rng,8.)
        B=local_basis(z0,2,args.seed)
        base=decoder(z0)[0]
        projector_rgb=image_u8(base)
        initframe=proj.capture(cam,projector_rgb,args.settle,args.samples)
        offset=f.observe(initframe)-f.predict(screen_features(projector_rgb,f.grid))[0]
        # Offset corrects the constant forward-model mismatch at the initial face.
        a=np.zeros(2,np.float64);radius=8.;last_print=time.monotonic()
        floor=max(.012,3.*f.metrics['test_rmse'])
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        i=0
        try:
            while True:
                z=z0+B[0]*a[0]+B[1]*a[1]
                z=z*radius/(np.linalg.norm(z)+1e-10)
                rgb=decoder(z)[0]
                image=image_u8(rgb)
                frame=proj.capture(cam,image,args.settle,args.samples)
                y=f.observe(frame)
                predicted=f.predict(screen_features(image,f.grid))[0]+offset
                delta=(y-predicted).reshape(f.side,f.side)
                rms=float(np.sqrt(np.mean(delta*delta)))
                amplitude=float(np.clip((rms-floor)/max(.025,4*floor),0,1))
                xx,yy=np.meshgrid(np.linspace(-1,1,f.side),np.linspace(-1,1,f.side))
                # Signed spatial residual, not the raw camera image, nudges local
                # identity coordinates. Small changes decay to the immutable face.
                drives=np.array([float(np.mean(xx*delta)),float(np.mean(yy*delta))])
                a=.91*a+args.gain*20*drives
                a=np.clip(a,-5,5)
                radius=.84*radius+.16*(8+37*amplitude)
                if time.monotonic()-last_print>1.0:
                    print(f'camera residual RMS {rms:.4f}  calibration floor {floor:.4f}  '
                          f'|z| {radius:.2f} ({shell_name(radius)}; NOT confidence)',flush=True)
                    last_print=time.monotonic()
                key=proj.last_key
                if key in (27,ord('q')):break
                if key==ord('r'):a[:]=0;radius=8
                if key==ord('s'):
                    cv2.imwrite(str(out/f'resonator_{i:06d}.png'),frame)
                i+=1
        except KeyboardInterrupt:
            pass
    finally:
        try:proj.close()
        except Exception:pass
        cam.close()


class ToyDecoder:
    """Explicitly synthetic, deterministic differentiable 6D image family."""
    def __init__(self):
        yy,xx=np.mgrid[0:96,0:96].astype(np.float32)
        basis=[]
        for i in range(6):
            cx=20+((i*33)%59);cy=22+((i*23)%51)
            basis.append(np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*(8+i*2)**2)))
        self.b=np.stack(basis)
    def __call__(self,z):
        z=np.asarray(z,np.float32)
        if z.ndim==1:z=z[None]
        vals=np.tanh(z[:,:6]*.8)
        gray=np.clip(.38+np.einsum('ni,ihw->nhw',vals,self.b)*.12,0,1)
        return np.repeat(gray[:,:,:,None],3,axis=3)


def simulate(args):
    """No hardware, ONNX, camera, or claim about physical reconstruction."""
    decoder=ToyDecoder()
    rng=np.random.default_rng(2026)
    f=MeasuredForward(grid=8,side=14,ridge=.15)
    def camera_model(x):
        frame=cv2.resize(np.asarray(x,np.float32).reshape(8,8),(14,14),interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(frame,(0,0),1.6).ravel()
    X=np.asarray([screen_features(pattern(rng,8,i),8) for i in range(250)])
    Y=np.stack([camera_model(x) for x in X])
    f.roi=(0,0,14,14)
    m=f.fit(X,Y)
    z=new_latent(np.random.default_rng(args.seed+555),8.)
    truth=decoder(z)[0]
    masks=mask_bank(8)
    obs=[f.predict(screen_features(masked_image(truth,masks[0][1])*255,8))[0]]
    solved=solve_latent(decoder,f,np.concatenate(obs),[masks[0][1]],seed=args.seed,
                        candidates=min(128,args.candidates),steps=min(10,args.steps))
    result={'simulation_only':True,'measured_forward_heldout':m,
            'initial_observation_rmse':solved['initial_rmse'],
            'final_observation_rmse':solved['observed_rmse'],
            'image_rmse':float(np.sqrt(np.mean((decoder(solved['latent'])[0]-truth)**2))),
            'note':'Synthetic forward and synthetic decoder; not physical evidence.'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'simulation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode',choices=['studio','calibrate','inverse','feedback','simulate'])
    p.add_argument('--model',help='Path to genuine SplatWorld ONNX decoder')
    p.add_argument('--download-model',action='store_true',help='Download pinned 7.2 MB SplatWorld decoder')
    p.add_argument('--camera',default='http://192.168.1.157:8080',help='IP Webcam URL or local camera number')
    p.add_argument('--calibration',help='Previously measured forward.npz (rig must be unchanged)')
    p.add_argument('--target-image',help='Physical inverse target outside face prior')
    p.add_argument('--anchor',help='High-resolution anchor photograph for real residue detail')
    p.add_argument('--screen',help='Screen resolution, e.g. 1920x1080 (Windows auto by default)')
    p.add_argument('--zone',type=int,default=420)
    p.add_argument('--grid',type=int,default=8)
    p.add_argument('--side',type=int,default=14)
    p.add_argument('--patterns',type=int,default=120)
    p.add_argument('--views',type=int,default=3)
    p.add_argument('--settle',type=float,default=.7)
    p.add_argument('--samples',type=int,default=2)
    p.add_argument('--seed',type=int,default=31)
    p.add_argument('--candidates',type=int,default=120)
    p.add_argument('--top',type=int,default=3)
    p.add_argument('--dimensions',type=int,default=8)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--output',default='runs/splatyou')
    args=p.parse_args()
    if args.mode=='simulate':return simulate(args)
    dec=decoder_for(args)
    if args.mode=='studio':return studio(args,dec)
    if args.mode=='calibrate':
        cam,proj=start_rig(args,dec)
        try:acquire(proj,cam,dec,grid=args.grid,side=args.side,count=args.patterns,
                    settle=args.settle,samples=args.samples,output=Path(args.output)/'calibration')
        finally:cam.close();proj.close()
        return
    if args.mode=='inverse':return inverse(args,dec)
    if args.mode=='feedback':return feedback(args,dec)

if __name__=='__main__':
    main()
