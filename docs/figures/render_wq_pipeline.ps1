$ErrorActionPreference = 'Stop'

# Renders the modeling-pipeline figure as a raster PNG for the manuscript.
#
# RESOLUTION.  The figure is line art and 7.5 pt type, which is the hardest case for a
# raster: every cell border is a hairline, and at low pixel density the rasteriser has to
# spread each one over a fraction of a pixel, which reads as graininess.  The fix is pixel
# density, not a different antialiasing setting -- MDPI asks for >=1000 dpi for line art
# and combination figures, so this renders at 1200 dpi.  At 6.396 in wide that is a
# 7675 px image, and the hairlines land on whole pixels.
#
# Do not drop this below 1000 dpi to save file size.  600 dpi was visibly grainy in the
# 7.5 pt detail text, which is already at the printed-size floor and has no margin to
# lose.  Cairo is the rasteriser (-Tpng:cairo); the plain -Tpng backend is worse on
# hairlines at any density.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$draftDir = Join-Path $root 'docs\report\draft\figures'
$name = 'wq_pipeline'

# Graphviz computes node positions and routes every directed edge.
$dot = (Get-Command dot -ErrorAction SilentlyContinue).Source
if (-not $dot) {
    $dot = 'C:\Program Files\Graphviz\bin\dot.exe'
}
if (-not (Test-Path -LiteralPath $dot)) {
    throw 'Graphviz dot.exe was not found. Install Graphviz and add its bin directory to PATH.'
}

$source = Join-Path $here "$name.dot"
$png    = Join-Path $here "$name.png"

& $dot -Tpng:cairo -Gdpi=1200 -o $png $source

# The canvas size is pinned by the WIDTH attributes in each band's spacer row, not
# computed. A width well over 6.39 in means a label line overran its column and silently
# widened the figure, which shrinks all of its type in print. Read the whole -Tplain
# stream before taking the first line: piping into Select-Object -First 1 closes the pipe
# early and makes Graphviz report "gvwrite_no_z problem" on an otherwise successful run.
$plainLines = @(& $dot -Tplain $source)
$plain = $plainLines[0] -split '\s+'
$w = [double]$plain[2]
$h = [double]$plain[3]
if ([math]::Abs($w - 6.396) -gt 0.02) {
    Write-Warning ("width is {0:N4} in, expected about 6.396 - a line has overrun its column." -f $w)
}
if ($h -gt 5.5) {
    Write-Warning ("height is {0:N2} in, over the 5.5 in budget." -f $h)
}

Copy-Item -LiteralPath $png -Destination (Join-Path $draftDir "$name.png") -Force
$px = [int]($w * 1200)
$mb = (Get-Item $png).Length / 1MB
Write-Output ("{0}: {1:N4} x {2:N2} in, 1200 dpi = {3} px wide, {4:N1} MB" -f $name, $w, $h, $px, $mb)
Write-Output ("  -> {0}" -f $png)
Write-Output ("  -> {0}" -f (Join-Path $draftDir "$name.png"))
