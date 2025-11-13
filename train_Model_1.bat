@echo off
echo ============================================================
echo OPTIMIZED TRAINING - Model_1
echo ============================================================
echo.
pause

python train_segmentation.py ^
    --img-height 1536 ^
    --img-width 1536 ^
    --batch-size 2 ^
    --lr 2e-4 ^
    --epochs 150 ^
    --patience 15 ^
    --use-amp ^
    --output-dir models_1

echo.
echo ============================================================
echo OPTIMIZED TRAINING COMPLETE
echo ============================================================
echo.
echo Results saved to: models_1\training_history.json
echo Compare with models_1\training_history.json to see improvements!
pause
