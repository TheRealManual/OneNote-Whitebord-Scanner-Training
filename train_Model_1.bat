@echo off
echo ============================================================
echo OPTIMIZED TRAINING - Model_1
echo ============================================================
echo.
echo python train_segmentation.py 
echo    --data-dir dataset 
echo    --epochs 150 
echo    --batch-size 2 
echo    --lr 1e-4 
echo    --weight-decay 1e-4 
echo    --dice-weight 0.6 
echo    --focal-weight 0.4 
echo    --focal-alpha 0.25 
echo    --img-height 2560 
echo    --img-width 2560 
echo    --patience 15 
echo    --use-amp 
echo    --model-dir models_1
echo .
pause

python train_segmentation.py ^
    --data-dir dataset ^
    --epochs 150 ^
    --batch-size 2 ^
    --lr 1e-4 ^
    --weight-decay 1e-4 ^
    --dice-weight 0.6 ^
    --focal-weight 0.4 ^
    --focal-alpha 0.25 ^
    --img-height 2560 ^
    --img-width 2560 ^
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
