#!/usr/bin/env python3
"""Direct YOLO26n-seg controlled-image fruit segmentation. No SAM2 stage."""
import argparse,csv
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
VALID={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--input_dir',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--imgsz',type=int,default=1024); ap.add_argument('--conf',type=float,default=0.25); ap.add_argument('--iou',type=float,default=0.70); a=ap.parse_args()
    inp=Path(a.input_dir); out=Path(a.output_dir); md=out/'masks'; od=out/'overlays'; md.mkdir(parents=True,exist_ok=True); od.mkdir(parents=True,exist_ok=True); model=YOLO(a.model); rows=[]
    for p in sorted(inp.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID: continue
        r=model.predict(source=str(p),imgsz=a.imgsz,conf=a.conf,iou=a.iou,retina_masks=True,verbose=False)[0]
        if r.masks is None or r.boxes is None or len(r.masks.data)==0: rows.append({'image':p.name,'status':'no_mask'}); continue
        h,w=r.orig_shape; cand=[]
        for i,(m,c) in enumerate(zip(r.masks.data.detach().cpu().numpy(),r.boxes.conf.detach().cpu().numpy())):
            if m.shape!=(h,w): m=cv2.resize(m.astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST)
            mask=m>0.5; area=int(mask.sum())
            if area: cand.append((area,float(c),i,mask))
        if not cand: rows.append({'image':p.name,'status':'empty_masks'}); continue
        area,conf,idx,mask=sorted(cand,key=lambda x:(x[0],x[1]),reverse=True)[0]
        mp=md/f'{p.stem}.png'; cv2.imwrite(str(mp),mask.astype(np.uint8)*255)
        img=cv2.imread(str(p)); vis=img.copy(); ov=vis.copy(); ov[mask]=(0,255,0); vis=cv2.addWeighted(vis,.72,ov,.28,0); cnt,_=cv2.findContours(mask.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis,cnt,-1,(255,255,255),2); cv2.imwrite(str(od/f'{p.stem}_overlay.jpg'),vis)
        rows.append({'image':p.name,'status':'ok','confidence':conf,'mask_pixels':area,'mask_file':mp.name}); print(f'[OK] {p.name}: conf={conf:.3f}, area={area}')
    fp=out/'segmentation_summary.csv'; fields=sorted({k for r in rows for k in r});
    with fp.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
