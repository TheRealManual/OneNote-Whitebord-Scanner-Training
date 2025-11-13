@echo off
echo ============================================================
echo OPTIMIZED TRAINING - Model_2
echo ============================================================
echo.
pause

python train_segmentation.py ^
    --epochs 150 ^
    --batch-size 4 ^
    --lr 1e-4 ^
    --img-height 1024 ^
    --img-width 1024 ^
    --patience 20 ^
    --output-dir models_2 ^
    --use-amp

echo.
echo ============================================================
echo OPTIMIZED TRAINING COMPLETE
echo ============================================================
echo.
echo Results saved to: models_2\training_history.json
echo Compare with models_2\training_history.json to see improvements!
pause
