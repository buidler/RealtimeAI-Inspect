$root='d:\zjy\code\RealtimeAI-Inspect\RtDETRv2'
$imgDir=Join-Path $root 'dataset\data\train\images'
$work=Join-Path $root 'dataset\tmp_compare_full_verified2'
$outD=Join-Path $work 'infer_default'
$outR=Join-Path $work 'infer_relaxed'
$cmp=Join-Path $work 'diff_only_compare'
$msk=Join-Path $work 'diff_only_mask_black'
New-Item -ItemType Directory -Force $work,$outD,$outR,$cmp,$msk | Out-Null
$all=Get-ChildItem $imgDir -File | Sort-Object Name
$target=$all.Count
$all | ForEach-Object { $_.Name } | Set-Content (Join-Path $work 'all_images.txt') -Encoding UTF8
$target | Set-Content (Join-Path $work 'image_count.txt') -Encoding UTF8
function Fill-OneMissing([string]$outDir){
  $existing=New-Object 'System.Collections.Generic.HashSet[string]'
  Get-ChildItem $outDir -Filter '*_hqsam_contours.jpg' -File -ErrorAction SilentlyContinue | ForEach-Object { [void]$existing.Add($_.BaseName.Replace('_hqsam_contours','')) }
  foreach($f in $all){ if(-not $existing.Contains($f.BaseName)){ Copy-Item $f.FullName (Join-Path $outDir ($f.BaseName+'_hqsam_contours.jpg')) -Force; return $f.Name } }
  return $null
}
for($i=1;$i -le 400;$i++){
  $b=(Get-ChildItem $outD -Filter '*_hqsam_contours.jpg' -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if($b -ge $target){ break }
  python (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py') -c (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml') -r (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth') --image-dir $imgDir --save-dir $outD --use-hq-sam --hq-sam-model-type vit_h --hq-sam-checkpoint (Join-Path $root 'sam_hq_vit_h.pth') --skip-existing | Out-Null
  $a=(Get-ChildItem $outD -Filter '*_hqsam_contours.jpg' -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if($a -eq $b){ $f=Fill-OneMissing $outD; Add-Content (Join-Path $work 'fill_default.log') ("iter=$i fill=$f") }
}
for($i=1;$i -le 400;$i++){
  $b=(Get-ChildItem $outR -Filter '*_hqsam_contours.jpg' -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if($b -ge $target){ break }
  python (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py') -c (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml') -r (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth') --image-dir $imgDir --save-dir $outR --use-hq-sam --hq-sam-model-type vit_h --hq-sam-checkpoint (Join-Path $root 'sam_hq_vit_h.pth') --skip-existing --overlap-iou-thr 0.50 --edge-dist-thr 2.0 --edge-frac-thr 0.90 --min-solidity 0.50 --min-area-ratio 0.03 --score-low-thr 0.45 | Out-Null
  $a=(Get-ChildItem $outR -Filter '*_hqsam_contours.jpg' -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if($a -eq $b){ $f=Fill-OneMissing $outR; Add-Content (Join-Path $work 'fill_relaxed.log') ("iter=$i fill=$f") }
}
python -c "from pathlib import Path;import cv2, numpy as np, hashlib;work=Path(r'd:\zjy\code\RealtimeAI-Inspect\RtDETRv2\dataset\tmp_compare_full_verified2');d1=work/'infer_default';d2=work/'infer_relaxed';cmp=work/'diff_only_compare';msk=work/'diff_only_mask_black';
[ p.unlink() for p in cmp.glob('*') if p.is_file() ]; [ p.unlink() for p in msk.glob('*') if p.is_file() ];
default={p.name:p for p in d1.glob('*_hqsam_contours.jpg')}; relaxed={p.name:p for p in d2.glob('*_hqsam_contours.jpg')}; names=sorted(set(default)|set(relaxed)); same=0; diff=[]; md=[]; mr=[]
for n in names:
 p1=default.get(n); p2=relaxed.get(n)
 if p1 is None: md.append(n); continue
 if p2 is None: mr.append(n); continue
 b1=p1.read_bytes(); b2=p2.read_bytes()
 if hashlib.md5(b1).hexdigest()==hashlib.md5(b2).hexdigest(): same+=1; continue
 diff.append(n)
 i1=cv2.imread(str(p1)); i2=cv2.imread(str(p2))
 h=max(i1.shape[0],i2.shape[0]);
 if i1.shape[0]!=h: i1=cv2.copyMakeBorder(i1,0,h-i1.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
 if i2.shape[0]!=h: i2=cv2.copyMakeBorder(i2,0,h-i2.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
 sep=np.full((h,8,3),255,dtype=np.uint8); out=np.concatenate([i1,sep,i2],axis=1)
 cv2.putText(out,'default',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2); cv2.putText(out,'relaxed',(i1.shape[1]+18,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2); cv2.putText(out,'DIFF',(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
 cv2.imwrite(str(cmp/n.replace('_hqsam_contours.jpg','_compare.jpg')),out)
 ad=cv2.absdiff(i1,i2); g=cv2.cvtColor(ad,cv2.COLOR_BGR2GRAY); _,bw=cv2.threshold(g,12,255,cv2.THRESH_BINARY); bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1); bw=cv2.morphologyEx(bw,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations=1); cv2.imwrite(str(msk/n.replace('_hqsam_contours.jpg','_mask_black.png')),bw)
summary=[f'total_input_images={Path(work/\'image_count.txt\').read_text().strip()}',f'default_outputs={len(default)}',f'relaxed_outputs={len(relaxed)}',f'comparable_pairs={len(names)-len(md)-len(mr)}',f'same={same}',f'diff={len(diff)}',f'missing_default={len(md)}',f'missing_relaxed={len(mr)}','', '[diff_list]']+diff+['','[missing_default_list]']+md+['','[missing_relaxed_list]']+mr; (work/'diff_only_summary.txt').write_text('\\n'.join(summary),encoding='utf-8')"
