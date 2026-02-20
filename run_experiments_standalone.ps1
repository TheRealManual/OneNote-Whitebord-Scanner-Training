Set-Location $PSScriptRoot
cmd /c scripts\run_experiments.bat 2>&1 | Tee-Object -FilePath "..\SegmentationResearchPaper\experiments\experiment_log.txt"
Write-Host "`nExperiments finished. Press any key to close."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
