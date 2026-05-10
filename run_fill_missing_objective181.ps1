$root='d:\zjy\code\RealtimeAI-Inspect\RtDETRv2'
$imgDir=Join-Path $root 'dataset\data\train\images'
$work=Join-Path $root 'dataset\tmp_compare_objective181'
$outD=Join-Path $work 'infer_default'
$outR=Join-Path $work 'infer_relaxed'
$all=Get-ChildItem $imgDir -File | Sort-Object Name
function HasOut([string]$outDir,[string]$stem){ Test-Path (Join-Path $outDir ($stem + '_hqsam_contours.jpg')) }
foreach($f in $all){
  if(-not (HasOut $outD $f.BaseName)){
    python (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py') -c (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml') -r (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth') -f $f.FullName --save-dir $outD --use-hq-sam --hq-sam-model-type vit_h --hq-sam-checkpoint (Join-Path $root 'sam_hq_vit_h.pth') | Out-Null
  }
}
foreach($f in $all){
  if(-not (HasOut $outR $f.BaseName)){
    python (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py') -c (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml') -r (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth') -f $f.FullName --save-dir $outR --use-hq-sam --hq-sam-model-type vit_h --hq-sam-checkpoint (Join-Path $root 'sam_hq_vit_h.pth') --overlap-iou-thr 0.50 --edge-dist-thr 2.0 --edge-frac-thr 0.90 --min-solidity 0.50 --min-area-ratio 0.03 --score-low-thr 0.45 | Out-Null
  }
}
