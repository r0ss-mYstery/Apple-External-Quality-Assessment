#!/usr/bin/env python3
"""Derive 65th-98th percentile CIELAB core band from direct YOLO26n-seg masks."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
DEFAULT={'Fuji':{'s_min':55,'v_min':40,'v_max':240},'GoldenDelicious':{'s_min':35,'v_min':60,'v_max':255}}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cultivar',required=True,choices=['Fuji','GoldenDelicious']); ap.add_argument('--image_dir',required=True); ap.add_argument('--mask_dir',required=True); ap.add_argument('--out_json',required=True); ap.add_argument('--p_lo',type=float,default=.65); ap.add_argument('--p_hi',type=float,default=.98); a=ap.parse_args()
    vals=[]; used=0; gate=DEFAULT[a.cultivar]; imd=Path(a.image_dir); md=Path(a.mask_dir)
    for p in sorted(imd.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID: continue
        mp=md/f'{p.stem}.png';
        if not mp.exists(): continue
        img=cv2.imread(str(p)); m=cv2.imread(str(mp),0)
        if img is None or m is None: continue
        if m.shape!=img.shape[:2]: m=cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=m>127; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); S,V=hsv[:,:,1],hsv[:,:,2]; valid=mask&(S>=gate['s_min'])&(V>=gate['v_min'])&(V<=gate['v_max'])
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); C=lab[:,:,1].astype(np.float32)-128 if a.cultivar=='Fuji' else lab[:,:,2].astype(np.float32)-128
        if np.any(valid): vals.append(C[valid]); used+=1
    if not vals: raise RuntimeError('No valid fruit pixels collected')
    x=np.concatenate(vals); lo=float(np.quantile(x,a.p_lo)); hi=float(np.quantile(x,a.p_hi)); ch='a_star' if a.cultivar=='Fuji' else 'b_star'
    result={'cultivar':a.cultivar,'channel':ch,'p_lo':a.p_lo,'p_hi':a.p_hi,'core_band':[lo,hi],'initial_sv_gate':gate,'images_used':used,'pixels_used':int(x.size)}
    o=Path(a.out_json); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
