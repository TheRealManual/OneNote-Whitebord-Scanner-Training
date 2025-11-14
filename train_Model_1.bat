@echo off
echo ============================================================
echo OPTIMIZED TRAINING - Model_1
echo ============================================================
echo.
echo python train_segmentation.py 
echo    --data-dir dataset 
echo    --epochs 200 
echo    --batch-size 2 
echo    --lr 1e-4 
echo    --weight-decay 3e-4 
echo    --dice-weight 0.4 
echo    --focal-weight 0.6 
echo    --focal-alpha 0.75 
echo    --img-height 1536 
echo    --img-width 1536 
echo    --patience 20 
echo    --use-amp 
echo    --model-dir models_1
echo .
pause

python train_segmentation.py ^
    --data-dir dataset ^
    --epochs 200 ^
    --batch-size 2 ^
    --lr 1e-4 ^
    --weight-decay 3e-4 ^
    --dice-weight 0.4 ^
    --focal-weight 0.6 ^
    --focal-alpha 0.75 ^
    --img-height 1536 ^
    --img-width 1536 ^
    --patience 20 ^
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
