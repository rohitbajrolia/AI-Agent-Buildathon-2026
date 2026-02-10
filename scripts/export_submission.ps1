$ErrorActionPreference = 'Stop'

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutDir = Join-Path $RootDir 'dist'

function Require-Cmd([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $cmd) {
    throw "Missing dependency: $Name"
  }
}

Require-Cmd pandoc
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Convert-Doc([string]$InFile, [string]$OutBase) {
  $InPath = Join-Path $RootDir $InFile
  if (-not (Test-Path $InPath)) {
    Write-Warning "Skip (not found): $InFile"
    return
  }

  Write-Host "Exporting: $InFile"

  $HtmlPath = Join-Path $OutDir ($OutBase + '.html')
  pandoc $InPath --from gfm --toc --standalone -o $HtmlPath

  $DocxPath = Join-Path $OutDir ($OutBase + '.docx')
  pandoc $InPath --from gfm --toc --standalone -o $DocxPath

  $wk = Get-Command wkhtmltopdf -ErrorAction SilentlyContinue
  if ($wk) {
    $PdfPath = Join-Path $OutDir ($OutBase + '.pdf')
    try {
      pandoc $InPath --from gfm --toc --standalone --pdf-engine=wkhtmltopdf -o $PdfPath
    } catch {
      Write-Warning "PDF export failed; DOCX was generated successfully."
    }
  }
}

Convert-Doc 'BUILDATHON_SUBMISSION.md' 'Coverage_Concierge_Proposal'
Convert-Doc 'PROJECT_BRIEF.md' 'Coverage_Concierge_Project_Brief'
Convert-Doc 'PITCH_90_SECONDS.md' 'Coverage_Concierge_Pitch_90s'

Write-Host ""
Write-Host "Done. Outputs are in: $OutDir"
Write-Host "If you want PDF export, install wkhtmltopdf or use Word/Google Docs to save the DOCX as PDF."