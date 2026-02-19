@echo off
REM ============================================================================
REM Experiment Runner for Whiteboard Segmentation Research Paper
REM
REM Runs the full experiment matrix:
REM   1. Loss ablation study: 5 losses × 3 seeds × 1536 resolution
REM   2. Resolution study:   2 resolutions × 3 seeds × best loss
REM   3. Classical baseline
REM
REM All outputs go to ../SegmentationResearchPaper/experiments/
REM ============================================================================

set RESULTS_DIR=..\SegmentationResearchPaper\experiments
set SPLITS_CFG=..\SegmentationResearchPaper\configs\test_splits.json

echo ============================================================
echo WHITEBOARD SEGMENTATION - FULL EXPERIMENT SUITE
echo ============================================================
echo Results: %RESULTS_DIR%
echo.

REM ============================================================================
REM 1. LOSS ABLATION STUDY — 5 losses × 3 seeds at 1536×1152
REM ============================================================================

echo.
echo [1/3] LOSS ABLATION STUDY
echo ============================================================

for %%L in (ce dice focal dice_focal tversky) do (
    for %%S in (42 123 7) do (
        echo.
        echo --- Loss=%%L  Seed=%%S ---
        set PYTHONHASHSEED=%%S
        python train_segmentation.py ^
            --epochs 100 ^
            --batch-size 2 ^
            --lr 0.0002 ^
            --loss %%L ^
            --seed %%S ^
            --deterministic ^
            --img-height 1152 ^
            --img-width 1536 ^
            --test-split-config %SPLITS_CFG% ^
            --output-dir "%RESULTS_DIR%\loss_study\%%L_seed%%S" ^
            --use-amp
        if %ERRORLEVEL% NEQ 0 echo FAILED: %%L seed=%%S
    )
)

REM ============================================================================
REM 2. RESOLUTION STUDY — 768×1024 vs 1152×1536 with dice_focal × 3 seeds
REM ============================================================================

echo.
echo [2/3] RESOLUTION STUDY
echo ============================================================

for %%H in (768 1152) do (
    if %%H==768 (set W=1024) else (set W=1536)
    for %%S in (42 123 7) do (
        echo.
        echo --- Resolution=%%Hx!W!  Seed=%%S ---
        set PYTHONHASHSEED=%%S
        python train_segmentation.py ^
            --epochs 100 ^
            --batch-size 2 ^
            --lr 0.0002 ^
            --loss dice_focal ^
            --seed %%S ^
            --deterministic ^
            --img-height %%H ^
            --img-width !W! ^
            --test-split-config %SPLITS_CFG% ^
            --output-dir "%RESULTS_DIR%\resolution_study\%%Hx!W!_seed%%S" ^
            --use-amp
        if %ERRORLEVEL% NEQ 0 echo FAILED: %%Hx!W! seed=%%S
    )
)

REM ============================================================================
REM 3. CLASSICAL BASELINE
REM ============================================================================

echo.
echo [3/3] CLASSICAL BASELINE
echo ============================================================

python scripts/classical_baseline.py ^
    --images-dir dataset\images ^
    --masks-dir dataset\masks ^
    --output-dir "%RESULTS_DIR%" ^
    --test-split-config %SPLITS_CFG% ^
    --test-split both
if %ERRORLEVEL% NEQ 0 echo FAILED: classical baseline

echo.
echo ============================================================
echo ALL EXPERIMENTS COMPLETE
echo Results at: %RESULTS_DIR%
echo ============================================================
