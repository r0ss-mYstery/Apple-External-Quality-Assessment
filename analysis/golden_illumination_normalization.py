#!/usr/bin/env python3
"""Golden exposure-normalization sensitivity analysis: target median V=116, max gain=2, brighten only."""
import argparse,csv
from pathlib import Path
import cv2,numpy as np
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def norm(img,target=116.,max_gain=2.):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv);med=float(np.median(v));g=1. if med<=0 or med>=target else min(target/med,max_gain);vn=np.clip(v.astype(np.float32)*g,0,255).astype(np.uint8);return cv2.cvtColor(cv2.merge((h,s,vn)),cv2.COLOR_HSV2BGR),med,g,float(np.median(vn))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input_dir',required=True);ap.add_argument('--output_dir',required=True);ap.add_argument('--target_v',type=float,default=116.);ap.add_argument('--max_gain',type=float,default=2.);a=ap.parse_args();inp=Path(a.input_dir);out=Path(a.output_dir);nd=out/'normalized_images';nd.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(inp.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID:continue
        img=cv2.imread(str(p));
        if img is None:continue
        ni,b,g,af=norm(img,a.target_v,a.max_gain);op=nd/f'{p.stem}_normalized{p.suffix.lower()}';cv2.imwrite(str(op),ni);rows.append({'image':p.name,'original_median_v':b,'gain':g,'normalized_median_v':af,'brightened':int(g>1+1e-9)});print(f'[OK] {p.name}: {b:.1f}->{af:.1f}, gain={g:.3f}')
    fp=out/'illumination_normalization_metrics.csv';
    with fp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['image','original_median_v','gain','normalized_median_v','brightened']);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
