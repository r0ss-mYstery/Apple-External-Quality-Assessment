#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2, numpy as np
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def srgb_to_linear(c):
    a=0.055; c=np.clip(c,0,1).astype(np.float32); o=np.empty_like(c); lo=c<=0.04045; o[lo]=c[lo]/12.92; o[~lo]=((c[~lo]+a)/(1+a))**2.4; return o
def linear_to_srgb(c):
    a=0.055; c=np.clip(c,0,1).astype(np.float32); o=np.empty_like(c); lo=c<=0.0031308; o[lo]=12.92*c[lo]; o[~lo]=(1+a)*np.power(c[~lo],1/2.4)-a; return o
def apply_wb(img,g):
    rgb=img[:,:,::-1].astype(np.float32)/255.; lin=srgb_to_linear(rgb)*g[None,None,:]; return (np.clip(linear_to_srgb(lin),0,1)*255+0.5).astype(np.uint8)[:,:,::-1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_dir',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--gains',required=True); ap.add_argument('--jpeg_quality',type=int,default=95); a=ap.parse_args()
    g=np.array([float(x) for x in a.gains.split(',')],np.float32)
    if g.shape!=(3,): raise ValueError('--gains must be R,G,B')
    inp=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for p in sorted(inp.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID: continue
        img=cv2.imread(str(p));
        if img is None: continue
        wb=apply_wb(img,g); op=out/(p.stem+'_wb'+('.png' if p.suffix.lower()=='.png' else '.jpg'))
        cv2.imwrite(str(op),wb,[int(cv2.IMWRITE_JPEG_QUALITY),a.jpeg_quality] if op.suffix=='.jpg' else [])
        print('[OK]',p.name,'->',op.name)
if __name__=='__main__': main()
