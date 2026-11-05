#!/usr/bin/env python3
"""Recompute Golden coverage using SAME masks before/after normalization and unchanged fixed HSV rule."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
EXT=('.png','.jpg','.jpeg','.tif','.tiff')
def find(folder,stem,norm=False):
    names=[f'{stem}_normalized{e}' for e in EXT] if norm else []
    names += [f'{stem}{e}' for e in EXT]
    return next((folder/n for n in names if (folder/n).exists()),None)
def masks(root,stem):
    p=root/f'{stem}.png'
    if p.exists():return [p]
    d=root/stem;return sorted(d.glob('*.png')) if d.is_dir() else []
def coverage(img,m,r):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);H,S,V=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2];hm=np.zeros_like(H,dtype=bool)
    for lo,hi in r['hue_ranges']:hm|=(H>=lo)&(H<=hi)
    f=m.astype(bool);t=f&hm&(S>=r['s_min'])&(V>=r['v_min'])&(V<=r['v_max']);return 100*int(t.sum())/int(f.sum())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original_dir',required=True);ap.add_argument('--normalized_dir',required=True);ap.add_argument('--mask_dir',required=True);ap.add_argument('--config',required=True);ap.add_argument('--output_csv',required=True);a=ap.parse_args();od=Path(a.original_dir);nd=Path(a.normalized_dir);mr=Path(a.mask_dir);r=json.loads(Path(a.config).read_text())['GoldenDelicious'];rows=[]
    for p in sorted(od.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXT:continue
        npth=find(nd,p.stem,True);mps=masks(mr,p.stem)
        if npth is None or not mps:continue
        oi=cv2.imread(str(p));ni=cv2.imread(str(npth))
        for mp in mps:
            m=cv2.imread(str(mp),0);m=cv2.resize(m,(oi.shape[1],oi.shape[0]),interpolation=cv2.INTER_NEAREST) if m.shape!=oi.shape[:2] else m;m=m>127;oc=coverage(oi,m,r);nc=coverage(ni,m,r);rows.append({'image':p.name,'mask':mp.name,'original_coverage_pct':oc,'normalized_coverage_pct':nc,'change_pp':nc-oc});print(f'[OK] {p.stem}/{mp.stem}: {oc:.2f}->{nc:.2f} ({nc-oc:+.2f} pp)')
    fp=Path(a.output_csv);fp.parent.mkdir(parents=True,exist_ok=True)
    with fp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['image','mask','original_coverage_pct','normalized_coverage_pct','change_pp']);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
