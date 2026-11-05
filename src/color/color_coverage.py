#!/usr/bin/env python3
"""Apply fixed final HSV rules to one or more fruit masks and report visible-surface coverage."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def masks(root,stem):
    p=root/f'{stem}.png'
    if p.exists():return [p]
    d=root/stem
    return sorted(d.glob('*.png')) if d.is_dir() else []
def cov(img,m,rule):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);H,S,V=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2];hm=np.zeros_like(H,dtype=bool)
    for lo,hi in rule['hue_ranges']:hm|=(H>=lo)&(H<=hi)
    fruit=m.astype(bool); target=fruit&hm&(S>=rule['s_min'])&(V>=rule['v_min'])&(V<=rule['v_max']);n=int(fruit.sum());return (np.nan if n==0 else 100*int(target.sum())/n),n,int(target.sum()),target
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cultivar',required=True,choices=['Fuji','GoldenDelicious']);ap.add_argument('--image_dir',required=True);ap.add_argument('--mask_dir',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output_dir',required=True);a=ap.parse_args()
    rule=json.loads(Path(a.config).read_text())[a.cultivar]; imd=Path(a.image_dir); mr=Path(a.mask_dir); out=Path(a.output_dir); od=out/'overlays';od.mkdir(parents=True,exist_ok=True); rows=[]
    for p in sorted(imd.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID:continue
        img=cv2.imread(str(p));
        if img is None:continue
        for mp in masks(mr,p.stem):
            m=cv2.imread(str(mp),0);m=cv2.resize(m,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST) if m.shape!=img.shape[:2] else m; m=m>127; c,n,nt,t=cov(img,m,rule); rows.append({'image':p.name,'mask':mp.name,'cultivar':a.cultivar,'fruit_pixels':n,'target_color_pixels':nt,'visible_color_coverage_pct':c}); print(f'[OK] {p.name}/{mp.name}: {c:.2f}%')
            vis=img.copy();vis[~m]=(vis[~m]*.2).astype(np.uint8);ov=vis.copy();ov[t]=(0,0,255) if a.cultivar=='Fuji' else (0,255,255);vis=cv2.addWeighted(vis,.72,ov,.28,0);cv2.imwrite(str(od/f'{p.stem}_{mp.stem}_coverage.png'),vis)
    fp=out/'color_coverage.csv';
    with fp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['image','mask','cultivar','fruit_pixels','target_color_pixels','visible_color_coverage_pct']);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
