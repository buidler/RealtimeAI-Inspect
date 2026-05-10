$ErrorActionPreference = 'Continue'
$root = 'd:\zjy\code\RealtimeAI-Inspect\RtDETRv2'
$imgDir = Join-Path $root 'dataset\data\train\images'
$work = Join-Path $root 'dataset\tmp_compare_full_verified2'
$outD = Join-Path $work 'infer_default'
$outR = Join-Path $work 'infer_relaxed'
$chunkRoot = Join-Path $work 'chunks'
$diffCmp = Join-Path $work 'diff_only_compare'
$diffMsk = Join-Path $work 'diff_only_mask_black'

if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $outD,$outR,$chunkRoot,$diffCmp,$diffMsk | Out-Null

$images = Get-ChildItem $imgDir -File | Sort-Object Name
$images | ForEach-Object { $_.Name } | Set-Content (Join-Path $work 'all_images.txt') -Encoding UTF8
$images.Count | Set-Content (Join-Path $work 'image_count.txt') -Encoding UTF8

function Ensure-ChunkDir([array]$chunk, [string]$chunkDir) {
  if (Test-Path $chunkDir) { Remove-Item $chunkDir -Recurse -Force }
  New-Item -ItemType Directory -Force $chunkDir | Out-Null
  foreach($f in $chunk){ Copy-Item $f.FullName (Join-Path $chunkDir $f.Name) -Force }
}

function Run-InferChunk([string]$chunkDir, [string]$saveDir, [bool]$relaxed){
  $args = @(
    (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py'),
    '-c', (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml'),
    '-r', (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth'),
    '--image-dir', $chunkDir,
    '--save-dir', $saveDir,
    '--use-hq-sam', '--hq-sam-model-type', 'vit_h', '--hq-sam-checkpoint', (Join-Path $root 'sam_hq_vit_h.pth'),
    '--skip-existing'
  )
  if($relaxed){
    $args += @('--overlap-iou-thr','0.50','--edge-dist-thr','2.0','--edge-frac-thr','0.90','--min-solidity','0.50','--min-area-ratio','0.03','--score-low-thr','0.45')
  }
  & python @args | Out-Null
  return $LASTEXITCODE
}

function Retry-Missing([array]$chunk, [string]$saveDir, [bool]$relaxed){
  foreach($f in $chunk){
    $outFile = Join-Path $saveDir ($f.BaseName + '_hqsam_contours.jpg')
    if(Test-Path $outFile){ continue }
    $args = @(
      (Join-Path $root 'references\deploy\rtdetrv2_hqsam_infer_v2.py'),
      '-c', (Join-Path $root 'configs\rtdetrv2\rtdetrv2_fiber.yml'),
      '-r', (Join-Path $root 'output\rtdetrv2_fiber_4\best.pth'),
      '-f', $f.FullName,
      '--save-dir', $saveDir,
      '--use-hq-sam', '--hq-sam-model-type', 'vit_h', '--hq-sam-checkpoint', (Join-Path $root 'sam_hq_vit_h.pth')
    )
    if($relaxed){
      $args += @('--overlap-iou-thr','0.50','--edge-dist-thr','2.0','--edge-frac-thr','0.90','--min-solidity','0.50','--min-area-ratio','0.03','--score-low-thr','0.45')
    }
    & python @args | Out-Null
    if($LASTEXITCODE -ne 0 -or -not (Test-Path $outFile)){
      Copy-Item $f.FullName $outFile -Force
      Add-Content (Join-Path $work 'fallback_used.txt') ((Get-Date).ToString('s') + ' ' + $outFile)
    }
  }
}

$chunkSize = 20
for($i=0; $i -lt $images.Count; $i += $chunkSize){
  $end = [Math]::Min($i + $chunkSize - 1, $images.Count - 1)
  $chunk = $images[$i..$end]
  $chunkName = ('chunk_{0:D3}' -f ([int]($i/$chunkSize)+1))
  $chunkDir = Join-Path $chunkRoot $chunkName
  Ensure-ChunkDir $chunk $chunkDir

  $codeD = Run-InferChunk $chunkDir $outD $false
  Retry-Missing $chunk $outD $false

  $codeR = Run-InferChunk $chunkDir $outR $true
  Retry-Missing $chunk $outR $true

  Add-Content (Join-Path $work 'chunk_status.txt') ("$chunkName default_exit=$codeD relaxed_exit=$codeR")
}
