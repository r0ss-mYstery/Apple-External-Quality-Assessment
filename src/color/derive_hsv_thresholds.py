#!/usr/bin/env python3
"""Derive HSV criteria from CIELAB core pixels: H 2-98%, S10%, V5-99%."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def signed_intervals(lo,hi):
    if lo>=0 and hi>=0:return [[int(round(lo)),int(round(hi))]]
    if lo<=0 and hi<=0:return [[int(round(180+lo)),int(round(180+hi))]]
    return [[0,int(round(hi))],[int(round(180+lo)),180]]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--band_json',required=True); ap.add_argument('--image_dir',required=True); ap.add_argument('--mask_dir',required=True); ap.add_argument('--out_json',required=True); a=ap.parse_args()
    b=json.loads(Path(a.band_json).read_text()); cult=b['cultivar']; lo,hi=b['core_band']; g=b['initial_sv_gate']; Hs=[];Ss=[];Vs=[]; imd=Path(a.image_dir); md=Path(a.mask_dir)
    for p in sorted(imd.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID: continue
        mp=md/f'{p.stem}.png';
        if not mp.exists():continue
        img=cv2.imread(str(p)); m=cv2.imread(str(mp),0)
        if img is None or m is None:continue
        if m.shape!=img.shape[:2]:m=cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=m>127; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); H,S,V=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]; valid=mask&(S>=g['s_min'])&(V>=g['v_min'])&(V<=g['v_max'])
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); C=lab[:,:,1].astype(np.float32)-128 if b['channel']=='a_star' else lab[:,:,2].astype(np.float32)-128; core=valid&(C>=lo)&(C<=hi)
        if np.any(core): Hs.append(H[core].astype(np.float32)); Ss.append(S[core].astype(np.float32)); Vs.append(V[core].astype(np.float32))
    if not Hs:raise RuntimeError('No core pixels found')
    H,S,V=np.concatenate(Hs),np.concatenate(Ss),np.concatenate(Vs); smin=int(round(np.quantile(S,.10))); vmin=int(round(np.quantile(V,.05))); vmax=int(round(np.quantile(V,.99)))
    if cult=='Fuji': hs=H.copy(); hs[hs>90]-=180; hr=signed_intervals(float(np.quantile(hs,.02)),float(np.quantile(hs,.98)))
    else: hr=[[int(round(np.quantile(H,.02))),int(round(np.quantile(H,.98)))]]
    r={'cultivar':cult,'hue_ranges':hr,'s_min':smin,'v_min':vmin,'v_max':vmax,'quantiles':{'h':[.02,.98],'s_min':.10,'v':[.05,.99]},'core_pixel_count':int(H.size)}
    o=Path(a.out_json);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
