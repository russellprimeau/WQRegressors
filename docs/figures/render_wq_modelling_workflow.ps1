$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $here 'wq_modelling_workflow.dot'
$output = Join-Path $here 'wq_modelling_workflow.png'

if (-not $env:WQ_MANUSCRIPT_ROOT) {
    throw 'WQ_MANUSCRIPT_ROOT is not set. Open wq-forecasting.code-workspace, or set it for this shell.'
}
$draftDir = Join-Path $env:WQ_MANUSCRIPT_ROOT 'figures'
if (-not (Test-Path -LiteralPath $draftDir)) {
    throw "Manuscript figures directory not found: $draftDir"
}
$draftOutput = Join-Path $draftDir 'wq_modelling_workflow.png'

# Graphviz computes node positions and routes every directed edge.
$dot = (Get-Command dot -ErrorAction SilentlyContinue).Source
if (-not $dot) {
    $dot = 'C:\Program Files\Graphviz\bin\dot.exe'
}
if (-not (Test-Path -LiteralPath $dot)) {
    throw 'Graphviz dot.exe was not found. Install Graphviz and add its bin directory to PATH.'
}
# Use Cairo at print resolution so type remains crisp after LaTeX scales the figure.
& $dot -Tpng:cairo -Gdpi=600 -o $output $source
Copy-Item -LiteralPath $output -Destination $draftOutput -Force
Write-Output "Updated $output and $draftOutput"
