#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2, numpy as np

def srgb_to_linear(c):
    a=0.055; c=np.clip(c,0,1).astype(np.float32); out=np.empty_like(c); low=c<=0.04045
    out[low]=c[low]/12.92; out[~low]=((c[~low]+a)/(1+a))**2.4; return out

def main():
    ap=argparse.ArgumentParser(description="Estimate linear-RGB WB gains from a neutral gray patch")
    ap.add_argument('--image',required=True); ap.add_argument('--out_json',default=None)
    ap.add_argument('--max_display_width',type=int,default=1600); ap.add_argument('--edge_trim',type=float,default=0.10)
    a=ap.parse_args(); p=Path(a.image); img=cv2.imread(str(p),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    scale=min(1.0,a.max_display_width/img.shape[1]); disp=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else img.copy()
    print('Select ONLY the interior of a neutral gray patch; avoid border/glare/shadow.')
    x,y,w,h=cv2.selectROI('Select neutral gray patch',disp,False,True); cv2.destroyAllWindows()
    if w<=0 or h<=0: raise RuntimeError('No ROI selected')
    x,y,w,h=[int(round(v/scale)) for v in (x,y,w,h)]; mx=int(w*a.edge_trim); my=int(h*a.edge_trim)
    patch=img[y+my:y+h-my,x+mx:x+w-mx]
    rgb=patch[:,:,::-1].astype(np.float32); measured=np.median(rgb.reshape(-1,3),axis=0)
    lin=srgb_to_linear(measured[None,:]/255.0)[0]; target=float(np.mean(lin)); gains=np.clip(target/(lin+1e-8),0.5,2.0)
    result={'source_image':str(p),'measured_srgb_R_G_B':[float(x) for x in measured],'linear_rgb_gains_R_G_B':[float(x) for x in gains]}
    print(json.dumps(result,indent=2)); print(f'--gains "{gains[0]:.6f},{gains[1]:.6f},{gains[2]:.6f}"')
    if a.out_json:
        o=Path(a.out_json); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__': main()
