$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$source = Join-Path $here 'wq_modelling_workflow.dot'
$output = Join-Path $here 'wq_modelling_workflow.png'
$draftOutput = Join-Path $root 'docs\report\draft\figures\wq_modelling_workflow.png'

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
